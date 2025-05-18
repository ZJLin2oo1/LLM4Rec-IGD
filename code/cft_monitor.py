import os
import sys
import math
import fire
import torch
import warnings
import numpy as np
import torch.nn.functional as F

from functools import partial
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments
)
from transformers.data.data_collator import PaddingStrategy
from datasets import Dataset as HFDataset

from dataset import IGDataset, CFTDataset
from trainer import DebiasedTrainer, DataCollatorForSeq2Seq_v2

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

def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    reference_item_path: str = "",
    output_dir: str = "./lora-alpaca",
    sample: int = -1,
    seed: int = 0,
    batch_size: int = 128,
    micro_batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 1e-4,
    cutoff_len: int = 512,
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: Optional[str] = None,
    local_rank: int = 0,
    deepspeed: str = "./deepspeed.json",
    category: str = "",
    K: int = 0,
    version: str = "base",
    beta: float = 0.0,
    alpha: float = 0.0
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

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    train_data = CFTDataset(train_file, tokenizer, cutoff_len, sample, seed, category, K)
    val_data = IGDataset(eval_file, tokenizer, cutoff_len, sample, category, K)
    print("Data Loading Complete")

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})

    trainer = DebiasedTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        alpha=alpha,
        beta=beta,
        args=TrainingArguments(
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
            ddp_find_unused_parameters=not ddp if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            logging_steps=100,
            disable_tqdm=False,
        ),
        data_collator=DataCollatorForSeq2Seq_v2(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("Saving Model...")
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)