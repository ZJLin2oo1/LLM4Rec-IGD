import os
import json
import fire
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessorList,
    TemperatureLogitsWarper
)

# Local modules for dataset and custom trie-based decoding logic
from dataset import IGDataset
from LogitProcesser import TrieLogitsProcessor
from trie import Trie

# Optionally import LoRA if available
try:
    from peft import PeftModel, PeftConfig
except ImportError:
    PeftModel = None
    PeftConfig = None

# Automatically choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

def modify_items(text, eos):
    """Append EOS token to the input item string."""
    return f'{text}\n{eos}'

def load_titles_to_trie_from_json(input_file, tokenizer, frequency_scale):
    """
    Load reference items and their frequencies from a JSON file into a Trie.

    Args:
        input_file (str): Path to JSON file mapping item titles to frequencies.
        tokenizer: HuggingFace tokenizer.
        frequency_scale (float): Total frequency value for normalization.

    Returns:
        Trie: Trie structure containing tokenized item titles.
    """
    trie = Trie(tokenizer=tokenizer, frequency_scale=frequency_scale)
    with open(input_file, 'r', encoding='utf-8') as file:
        item_frequency_dict = json.load(file)
    for title_name, frequency in tqdm(item_frequency_dict.items(), desc="Loading titles to Trie", mininterval=10.0):
        modify_title = modify_items(title_name, tokenizer.eos_token)
        trie.insert(modify_title, frequency)
    assert trie.frequency_scale == trie.total_frequency
    return trie

def load_model(base_model, use_lora=False):
    """
    Load a causal LM model, optionally with LoRA adaptation.

    Args:
        base_model (str): HuggingFace model name or path.
        use_lora (bool): Whether to load with LoRA adapters.

    Returns:
        torch.nn.Module: Loaded language model.
    """
    if use_lora:
        assert PeftModel is not None, "LORA is not installed"
        config = PeftConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(model, base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2="llama" in base_model.lower()
        )
    return model

def evaluate(model, tokenizer, encodings, rf_item_trie, temperature, length_penalty, alpha, num_beams=10, max_new_tokens=72):
    """
    Run generation with custom TrieLogitsProcessor for constrained decoding.

    Args:
        model: The language model.
        tokenizer: Tokenizer corresponding to the model.
        encodings (List[Dict]): Tokenized input samples.
        rf_item_trie (Trie): Trie used to constrain decoding and reweighting.
        temperature (float): Sampling temperature.
        length_penalty (float): Beam search length penalty.
        alpha (float): Weight for trie-based logit modification.
        num_beams (int): Beam size.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        Tuple[List[str], List[float], List[List[List[float]]]]:
            Generated sequences, final beam scores, and token-level scores per beam.
    """
    padded_encodings = tokenizer.pad(encodings, padding=True, return_tensors="pt")
    generation_config = GenerationConfig(
        num_beams=num_beams,
        length_penalty=length_penalty,
        num_return_sequences=num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens
    )
    with torch.no_grad():
        logits_processor = LogitsProcessorList([
            TemperatureLogitsWarper(temperature=temperature),
            TrieLogitsProcessor(tokenizer=tokenizer, num_beams=num_beams, trie=rf_item_trie, alpha=alpha)
        ])
        output = model.generate(
            input_ids=padded_encodings["input_ids"].to(device),
            attention_mask=padded_encodings["attention_mask"].to(device),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=logits_processor
        )
    input_lengths = padded_encodings["input_ids"].shape[1]
    s = output.sequences[:, input_lengths:]
    scores = output.sequences_scores.tolist()

    # Decode token-level scores per beam and per timestep
    batch_size = len(encodings)
    sequence_scores = [[[0 for _ in range(s.shape[1])] for _ in range(num_beams)] for _ in range(batch_size)]
    for b in range(batch_size):
        for beam in range(num_beams):
            idx = b * num_beams + beam
            for step in range(s.shape[1]):
                beam_index = output.beam_indices[idx][step]
                if beam_index != -1:
                    sequence_scores[b][beam][step] = output.scores[step][beam_index][s[idx][step]].item()

    # Decode final text outputs
    decoded = tokenizer.batch_decode(s, skip_special_tokens=True)
    stripped = [o.split("Response:")[-1] for o in decoded]
    grouped_outputs = [stripped[i * num_beams:(i + 1) * num_beams] for i in range(batch_size)]
    grouped_scores = [scores[i * num_beams:(i + 1) * num_beams] for i in range(batch_size)]
    return grouped_outputs, grouped_scores, sequence_scores

def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    reference_item_path: str = "",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 32,
    K: int = 0,
    seed: int = 0,
    temperature: float = 1.0,
    alpha: float = 0.0,
    length_penalty: float = 0.0,
    use_lora: bool = False
):
    """
    Main evaluation pipeline using trie-constrained generation for recommender-style tasks.

    Args are passed via CLI using Fire. Outputs are saved to result_json_data.
    """
    # Frequency and label mapping for different domains
    frequency_scale_dict = {
        "Books": 675262.0, "Video_Games": 195559.0, "Toys_and_Games": 111806.0,
        "CDs_and_Vinyl": 134436.0, "Sports_and_Outdoors": 160015.0
    }
    category_dict = {
        "Office_Products": "office products", "Books": "books", "steam": "games",
        "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games",
        "Video_Games": "video games", "Musical_Instruments": "music instruments",
        "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies",
        "Arts_Crafts_and_Sewing": "arts products", "STEAM": "games"
    }

    frequency_scale = frequency_scale_dict[category]
    category = category_dict[category]

    # Load model and tokenizer
    model = load_model(base_model, use_lora)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # Load prompt templates or prefix information (if needed)
    with open(info_file, 'r') as f:
        info_lines = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in f.readlines()]
        info = [f"""### Response: \n{line}""" for line in info_lines]

    # Load evaluation dataset and Trie
    val_dataset = IGDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    rf_item_trie = load_titles_to_trie_from_json(reference_item_path, tokenizer, frequency_scale)
    encodings = [val_dataset.__getitem__(i) for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.eval().to(device)

    # Divide into batches
    blocks = [encodings[i:i + batch_size] for i in range(0, len(encodings), batch_size)]

    outputs, scores, seq_scores = [], [], []
    for block in tqdm(blocks):
        output, score, seq_score = evaluate(model, tokenizer, block, rf_item_trie, temperature, length_penalty, alpha)
        outputs.extend(output)
        scores.extend(score)
        seq_scores.extend(seq_score)

    # Merge predictions back to original test data
    for i, item in enumerate(test_data):
        item["predict"] = outputs[i]
        item["predict_score"] = scores[i]
        item["predict_seq_score"] = seq_scores[i]
        item.pop("dedup", None)

    # Save prediction results
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

# Run main function with CLI arguments using Fire
if __name__ == '__main__':
    fire.Fire(main)
