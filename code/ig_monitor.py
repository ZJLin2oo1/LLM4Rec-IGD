import os
import sys
import math
import json
import warnings
from functools import partial
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import fire
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset, load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from trie import Trie
from dataset import IGDataset
from trainer import IGMonitorTrainer

# Set home directory and Hugging Face token
user_home = os.path.expanduser("~")
print(user_home)
HF_TOKEN = None

# Global constants
FREQ_SCALE_DICT = {
    "Books": 675262.0,
    "Video_Games": 195559.0,
    "Toys_and_Games": 111806.0,
    "CDs_and_Vinyl": 134436.0,
    "Sports_and_Outdoors": 160015.0
}

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
    "Movies": "movie"
}

# Learning rate scheduler with cosine decay and warmup
def _get_cosine_schedule_with_warmup_lr_lambda(current_step, *, num_warmup_steps, num_training_steps, num_cycles):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# Format title text with EOS token
def modify_items(text, eos):
    return f'{text}\n{eos}'

# Load item titles into a Trie with frequency info
def load_titles_to_trie(input_file, tokenizer):
    trie = Trie(tokenizer=tokenizer)
    with open(input_file, 'r') as file:
        id_title = json.load(file)
        for title, frequency in tqdm(id_title.items(), desc="Loading titles to Trie", mininterval=10.0):
            modify_title = modify_items(title, tokenizer.eos_token)
            trie.insert(modify_title, frequency)
    return trie

# Same as above, but supports frequency scaling (used for IG)
def load_titles_to_trie_from_json(input_file, tokenizer, frequency_scale):
    trie = Trie(tokenizer=tokenizer, frequency_scale=frequency_scale)
    with open(input_file, 'r', encoding='utf-8') as file:
        item_frequency_dict = json.load(file)
    for title_name, frequency in tqdm(item_frequency_dict.items(), desc="Loading titles to Trie", mininterval=10.0):
        modify_title = modify_items(title_name, tokenizer.eos_token)
        trie.insert(modify_title, frequency)
    print(trie.frequency_scale)
    print(trie.total_frequency)
    assert trie.frequency_scale == trie.total_frequency
    return trie

# Main training entry point
def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    reference_item_path="",
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    local_rank: int = 0,
    deepspeed: str = "./deepspeed.json",
    category: str = "",
    K: int = 0,
    beta: float = 1.0,
    version: str = "base"
):
    print("Training Start")
    assert base_model, "Please specify a --base_model"

    frequency_scale = FREQ_SCALE_DICT[category]
    category_str = category
    category = CATEGORY_DICT[category]

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps //= world_size
    os.environ["WANDB_DISABLED"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA: {torch.cuda.is_available()}")
    print(f"Beta: {beta}, Category: {category}")

    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Setup for distributed training
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps //= world_size

    os.environ["WANDB_DISABLED"] = "true"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    # Load base model and apply LoRA if large
    if any(size in base_model.lower() for size in ["7b", "8b", "13b"]):
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        from peft import LoraConfig, get_peft_model
        print("Using LoRA for fine-tuning")
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, token=HF_TOKEN, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build item trie with reference item frequencies
    rf_item_trie = load_titles_to_trie_from_json(reference_item_path, tokenizer=tokenizer, frequency_scale=frequency_scale)
    rf_item_trie.compute_information_gain()

    # Load training and validation data
    train_data = IGDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, seed=seed, category=category, K=K)
    val_data = IGDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len, sample=sample, category=category, K=K)
    print("LOAD DATA FINISHED")

    # Handle multi-GPU inference
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Convert data to HuggingFace format
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})

    # Initialize trainer
    trainer = IGMonitorTrainer(
        rf_item_trie=rf_item_trie,
        beta=beta,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            logging_steps=100,
            disable_tqdm=False,
        ),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        run_initial_eval=False
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    fire.Fire(train)