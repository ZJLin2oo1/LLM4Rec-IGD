import os
import json
import time
import torch
import fire
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from dataset import D3Dataset

# Ensure multiprocessing works correctly with vLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Constants
P = 998244353
MOD = int(1e9 + 9)

# Category mapping
CATEGORY_DICT = {
    "Office_Products": "office products", 
    "Books": "books", 
    "steam": "games", 
    "CDs_and_Vinyl": "musics", 
    "Toys_and_Games": "toys and games", 
    "Video_Games": "video games", 
    "Musical_Instruments": "music instruments", 
    "Sports_and_Outdoors": "sports and outdoors", 
    "Pet_Supplies": "pet supplies", 
    "Arts_Crafts_and_Sewing": "arts products", 
    "STEAM": "games"
}

# Modify input text format
def modify_items(text):
    """Add quotes and an end marker to the input text."""
    return f'{text}\n<|endoftext|>'

# Main function
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 16,
    K: int = 0,
    seed: int = 0,
    temperature: float = 1.0,
    length_penalty: float = 1.0,
    num_beams: int = 10,
    max_new_tokens: int = 72
):
    # Map category name
    category = CATEGORY_DICT.get(category, category)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model (vLLM)
    llm = LLM(model=base_model, tensor_parallel_size=torch.cuda.device_count())

    # Load dataset
    val_dataset = D3Dataset(
        train_file=test_data_path,
        tokenizer=tokenizer,
        max_len=2560,
        category=category,
        test=True,
        K=K,
        seed=seed
    )
    encodings = [val_dataset.__getitem__(i) for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    # Process info file
    with open(info_file, 'r') as f:
        info = [f"\"{line.split('\t')[0].strip()}\"\n" for line in f.readlines()]
        item_name = info
        info = [f"### Response:\n{_}" for _ in info]

    # --------------------
    # Evaluate function
    # --------------------
    def evaluate(
        encodings,
        temperature=1.0,
        num_beams=10,
        max_new_tokens=72,
        length_penalty=1.0
    ):
        input_texts = []
        input_token_lengths = []
        
        # Decode input text and get token lengths
        for e in encodings:
            input_ids = e["input_ids"]
            input_texts.append(tokenizer.decode(input_ids, skip_special_tokens=True))
            input_token_lengths.append(len(input_ids))
        
        # Create prompt format
        prompts = [{"prompt": text} for text in input_texts]
        
        # Beam search parameters
        beam_params = BeamSearchParams(
            beam_width=num_beams,
            max_tokens=max_new_tokens,
            temperature=temperature,
            length_penalty=length_penalty
        )
        
        # Perform beam search
        beam_outputs = llm.beam_search(prompts, beam_params)
        
        real_outputs = []
        real_scores = []
        sequence_scores = []
        
        for idx, output in enumerate(beam_outputs):
            batch_outputs = []
            batch_scores = []
            batch_seq_scores = []
            prompt_length = input_token_lengths[idx]
            
            # Process each beam result
            for seq in output.sequences:
                generated_tokens = seq.tokens[prompt_length:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_outputs.append(generated_text)
                
                # Accumulated logprob of generated tokens
                batch_scores.append(seq.cum_logprob)
                
                # Per-token logprob
                token_scores = []
                for i in range(prompt_length, len(seq.tokens)):
                    index_in_logprobs = i - prompt_length
                    if index_in_logprobs < len(seq.logprobs) and seq.tokens[i] in seq.logprobs[index_in_logprobs]:
                        token_scores.append(seq.logprobs[index_in_logprobs][seq.tokens[i]].logprob)
                    else:
                        token_scores.append(0.0)
                batch_seq_scores.append(token_scores)
            
            real_outputs.append(batch_outputs)
            real_scores.append(batch_scores)
            sequence_scores.append(batch_seq_scores)
        
        return real_outputs, real_scores, sequence_scores

    # --------------------
    # Start Evaluation
    # --------------------
    start_time = time.time()
    
    # Split encodings into batches
    new_encodings = [
        encodings[i * batch_size : (i + 1) * batch_size] 
        for i in range((len(encodings) + batch_size - 1) // batch_size)
    ]
    
    outputs = []
    scores = []
    seq_scores = []
    
    # Evaluate each batch
    for encodings in tqdm(new_encodings, desc="Evaluating"):
        output, score, seq_score = evaluate(
            encodings,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty
        )
        outputs.extend(output)
        scores.extend(score)
        seq_scores.extend(seq_score)
    
    # Total time taken
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

    # --------------------
    # Save Results
    # --------------------
    # Ensure result folder exists
    result_folder = os.path.dirname(result_json_data)
    if result_folder and not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Attach predictions to test data
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        test["predict_score"] = scores[i]
        test["predict_seq_score"] = seq_scores[i]

    # Remove redundant fields
    for test in test_data:
        test.pop("dedup", None)

    # Write to result file
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

# --------------------
# Entry Point
# --------------------
if __name__ == "__main__":
    fire.Fire(main)



    # def evaluate(
    #     encodings,
    #     temperature=1.0,
    #     num_beams=10,
    #     max_new_tokens=16,
    #     length_penalty=1.0
    # ):
    #     input_texts = tokenizer.batch_decode(
    #         [e["input_ids"] for e in encodings],
    #         skip_special_tokens=True
    #     )

    #     # 修正SamplingParams参数
    #     sampling_params = SamplingParams(
    #         # use_beam_search=True,
    #         temperature=temperature,
    #         top_p=1.0,
    #         max_tokens=max_new_tokens,
    #         n=num_beams,         
    #         logprobs=1,                
    #         skip_special_tokens=True,
    #     )

    #     # 生成
    #         # 生成
    #     outputs = llm.generate(input_texts, sampling_params, use_tqdm=True)
    #     real_outputs = []
    #     real_scores = []
    #     sequence_scores = []
        
    #     for output in outputs:
    #         batch_outputs = []
    #         batch_scores = []
    #         batch_seq_scores = []
            
    #         for beam_idx in range(len(output.outputs)):
    #             batch_outputs.append(output.outputs[beam_idx].text)
    #             batch_scores.append(output.outputs[beam_idx].cumulative_logprob)
                
    #             token_scores = []
    #             for token_id, logprobs_dict in zip(output.outputs[beam_idx].token_ids, output.outputs[beam_idx].logprobs):
    #                 if token_id in logprobs_dict:
    #                     token_scores.append(logprobs_dict[token_id].logprob)
    #                 else:
    #                     token_scores.append(0.0)
    #             batch_seq_scores.append(token_scores)
            
    #         real_outputs.append(batch_outputs[:num_beams])
    #         real_scores.append(batch_scores[:num_beams]) 
    #         sequence_scores.append(batch_seq_scores[:num_beams])
        
    #     return real_outputs, real_scores, sequence_scores