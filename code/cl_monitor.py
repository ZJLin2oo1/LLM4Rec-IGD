import os
import sys
from typing import List
import numpy as np
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np
import fire
import transformers
from datasets import load_dataset, concatenate_datasets
from torch.optim.lr_scheduler import LambdaLR

from trie import Trie
from tqdm import tqdm
import json

"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset as HFDataset
from my_dataset import IGDataset
import os
from datetime import datetime
from transformers import DataCollatorForSeq2Seq
user_home = os.path.expanduser("~")
print(user_home)
# 获取当前用户的 home 目录
def _get_cosine_schedule_with_warmup_lr_lambda(
        current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def modify_items(text):
    # 只在文本前后添加引号，并在结尾加换行符
    # result = f'"{text}"\n<|endoftext|>'
    result = f'{text}\n<|endoftext|>'
    return result


def load_titles_to_trie(input_file, tokenizer):
    """将标题和频率加载到Trie中，并在每个标题前加一个冒号"""
    trie = Trie(tokenizer=tokenizer)
    with open(input_file, 'r') as file:
        id_title = json.load(file)
        for title, frequency in tqdm(id_title.items(), desc="Loading titles to Trie", mininterval=10.0):
            modify_title = modify_items(title)
            trie.insert(modify_title, frequency)  # 插入修改后的标题
    return trie


def load_titles_to_trie_from_json(input_file, tokenizer, frequency_scale):
    """从 JSON 读取标题和频率，并加载到 Trie 中"""
    trie = Trie(tokenizer=tokenizer, frequency_scale=frequency_scale)

    # 读取 JSON 文件
    with open(input_file, 'r', encoding='utf-8') as file:
        item_frequency_dict = json.load(file)

    # 遍历 JSON 数据，将 item_title 插入 Trie
    for title_name, frequency in tqdm(item_frequency_dict.items(), desc="Loading titles to Trie", mininterval=10.0):
        modify_title = modify_items(title_name)  # 添加引号、换行符和终结符
        trie.insert(modify_title, frequency)  # 插入 Trie，权重为 frequency

    print(trie.frequency_scale)
    print(trie.total_frequency)
    # Sports_and_Outdoors: 160015.0
    # CDs_and_Vinyl: 134436.0
    # Toys_and_Games: 111806.0
    # Books: 675262.0
    # Video_Games: 195559.0
    assert trie.frequency_scale == trie.total_frequency
    return trie


import os
import json
import torch
from datetime import datetime
from transformers import Trainer


class IGMonitorTrainer(Trainer):
    def __init__(self, rf_item_trie, low_high_split, run_initial_eval=False, log_dir="./logs", beta=1.0, gamma=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rf_item_trie = rf_item_trie
        self.run_initial_eval = run_initial_eval
        self.log_dir = log_dir
        self.beta = beta  # 用于 zero IG token 的加权
        self.gamma = gamma  # 用于 low IG token 的加权
        self.low_high_split = low_high_split

        # 创建日志文件
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.jsonl")

        # 初始化日志文件
        with open(self.log_file, 'w') as f:
            f.write("")  # 创建空文件

        # 如果需要，在训练开始前运行一次评估
        if self.run_initial_eval:
            self.evaluate()

    def log_metrics(self, split, metrics, epoch=None):
        """将评估指标写入日志文件"""
        log_entry = {
            "split": split,
            "epoch": epoch if epoch is not None else self.state.epoch,
            "metrics": metrics
        }

        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

    def compute_loss(self, model, inputs, return_outputs=False):
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']

        # 移位，准备计算 CrossEntropy Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 计算 per-token Loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.size())

        # 忽略 -100 标签部分
        non_pad_mask = shift_labels != -100

        # 初始化 ig_weights
        ig_weights = torch.zeros_like(per_token_loss, dtype=torch.float32)  # 默认权重为 0

        # 初始化统计字典
        ig_groups = {
            "zero": {"total_loss": 0.0, "count": 0},
            "low": {"total_loss": 0.0, "count": 0},
            "high": {"total_loss": 0.0, "count": 0}
        }

        # 提前获取所有序列的 IG 值
        all_ig_lists = []
        for i, seq_labels in enumerate(shift_labels):
            valid_labels = seq_labels[non_pad_mask[i]].tolist()
            if valid_labels:
                ig_list = self.rf_item_trie.get_sequence_ig(valid_labels)
                if ig_list[-1] == float('-inf'):
                    ig_list = [0] * len(valid_labels)
                all_ig_lists.append(ig_list)
            else:
                all_ig_lists.append([])

        # 将所有 IG 值合并为一个 Tensor
        all_ig_tensors = [torch.tensor(ig_list, dtype=torch.float32, device=per_token_loss.device) for ig_list in all_ig_lists if ig_list]
        if all_ig_tensors:
            all_ig_tensor = torch.cat(all_ig_tensors)
        else:
            all_ig_tensor = torch.tensor([], dtype=torch.float32, device=per_token_loss.device)

        # zero_mask = all_ig_tensor == 0.0
        # high_mask = all_ig_tensor > 0.0
        #
        # ig_weights[non_pad_mask] = torch.where(zero_mask, self.beta, ig_weights[non_pad_mask])
        # ig_weights[non_pad_mask] = torch.where(high_mask, 1.0, ig_weights[non_pad_mask])

        zero_mask = all_ig_tensor == 0
        low_mask = (all_ig_tensor > 0) & (all_ig_tensor <= self.low_high_split)
        high_mask = all_ig_tensor > self.low_high_split

        ig_weights[non_pad_mask] = torch.where(zero_mask, self.beta, ig_weights[non_pad_mask])
        # ig_weights[non_pad_mask] = torch.where(low_mask, self.gamma, ig_weights[non_pad_mask])
        # ig_weights[non_pad_mask] = torch.where(high_mask, 1.0, ig_weights[non_pad_mask])
        ig_weights[non_pad_mask] = torch.where(low_mask, 1.0, ig_weights[non_pad_mask])
        ig_weights[non_pad_mask] = torch.where(high_mask, self.gamma, ig_weights[non_pad_mask])

        # 根据 IG 权重对 loss 进行分类统计
        for token_ig, loss_train in zip(all_ig_tensor, per_token_loss[non_pad_mask]):
            if token_ig == 0:
                group = "zero"
            elif 0 < token_ig <= self.low_high_split:
                group = "low"
            else:
                group = "high"

            ig_groups[group]["total_loss"] += loss_train.item()
            ig_groups[group]["count"] += 1

        new_weights = ig_weights[non_pad_mask]

        # 计算加权后的 loss
        valid_token_loss = (per_token_loss[non_pad_mask] * new_weights).sum() / new_weights.sum()

        # 记录 metrics
        state = "train" if self.is_in_train else "eval"
        self.log_metrics(state, {
            "ig_zero_loss": ig_groups["zero"]["total_loss"] / ig_groups["zero"]["count"] if ig_groups["zero"][
                                                                                                "count"] > 0 else float(
                'nan'),
            "ig_low_loss": ig_groups["low"]["total_loss"] / ig_groups["low"]["count"] if ig_groups["low"][
                                                                                             "count"] > 0 else float(
                'nan'),
            "ig_high_loss": ig_groups["high"]["total_loss"] / ig_groups["high"]["count"] if ig_groups["high"][
                                                                                                "count"] > 0 else float(
                'nan')
        }, epoch=self.state.epoch)

        # 返回 valid_token_loss 和 Outputs
        return (valid_token_loss, outputs) if return_outputs else valid_token_loss


    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        """自定义评估逻辑，用于计算 IG 相关统计"""
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        # 初始化分组统计
        ig_groups = {
            "zero": {"total_loss": 0.0, "count": 0},
            "low": {"total_loss": 0.0, "count": 0},
            "high": {"total_loss": 0.0, "count": 0}
        }

        for batch in dataloader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                logits = outputs.logits
                labels = batch['labels']

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_token_loss = per_token_loss.view(shift_labels.size())

                for seq_labels, seq_loss in zip(shift_labels, per_token_loss):
                    non_pad_mask = seq_labels != -100
                    valid_labels = seq_labels[non_pad_mask].cpu().tolist()
                    valid_losses = seq_loss[non_pad_mask].cpu().tolist()

                    if valid_labels:
                        ig_list = self.rf_item_trie.get_sequence_ig(valid_labels)
                        if ig_list[-1] == float('-inf'):
                            continue

                        for token_ig, loss_val in zip(ig_list, valid_losses):
                            if token_ig < 0:
                                continue
                            if token_ig == 0:
                                group = "zero"
                            elif 0 < token_ig <= 2:
                                group = "low"
                            else:
                                group = "high"

                            ig_groups[group]["total_loss"] += loss_val
                            ig_groups[group]["count"] += 1

        # 记录评估阶段的指标
        for group, stats in ig_groups.items():
            avg_loss = stats["total_loss"] / stats["count"] if stats["count"] > 0 else float('nan')
            output.metrics[f"{metric_key_prefix}_ig_{group}_loss"] = avg_loss
            output.metrics[f"{metric_key_prefix}_ig_{group}_count"] = stats["count"]
            print(f"IG Group: {group}, Avg Loss: {avg_loss:.4f}, Count: {stats['count']}")

        # 记录评估指标到日志文件
        self.log_metrics("eval", output.metrics, epoch=self.state.epoch)

        return output


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        train_file: str = "",
        eval_file: str = "",
        reference_item_path = "",
        output_dir: str = "./lora-alpaca",
        sample: int = -1,
        seed: int = 0,

        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 16,
        num_epochs: int = 3,
        # learning_rate: float = 1e-4,
        learning_rate: float = 1e-4,
        cutoff_len: int = 512,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

        local_rank: int = 0,
        deepspeed: str = "./deepspeed.json",
        category: str = "",
        K: int = 0,
        beta: float = 1.0,
        gamma: float = 1.0,
        version: str = "base"

):
    print(beta)
    print(gamma)
    # print(train_file)
    frequency_scale_dict = {"Books": 675262.0, "Video_Games": 195559.0, "Toys_and_Games": 111806.0,
                            "CDs_and_Vinyl": 134436.0, "Sports_and_Outdoors": 160015.0}
    # low_high_split_dict = {"Books": 4.759077, "Video_Games": 2.581968, "Toys_and_Games": 2.749671,
    #                         "CDs_and_Vinyl": 5.556205, "Sports_and_Outdoors": 0.105027}
    low_high_split_dict = {"Books": 4.759077, "Video_Games": 2.581968, "Toys_and_Games": 2.0,
                            "CDs_and_Vinyl": 5.556205, "Sports_and_Outdoors": 0.105027}
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games",
                     "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games",
                     "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors",
                     "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie"}
    print(category)
    frequency_scale = frequency_scale_dict[category]
    low_high_split = low_high_split_dict[category]
    category_str = category
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    # uses.environ["WANDB_LOG_MODEL"] = wandb_log_model
    os.environ["WANDB_DISABLED"] = "true"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())  # 检查是否可以使用 GPU
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    rf_item_trie = load_titles_to_trie_from_json(reference_item_path, tokenizer=tokenizer,
                                                 frequency_scale=frequency_scale)
    rf_item_trie.compute_information_gain()

    train_data = IGDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category, K = K)
    val_data = IGDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, category=category, K = K)

    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})

    trainer = IGMonitorTrainer(
        rf_item_trie=rf_item_trie,
        low_high_split=low_high_split,
        beta=beta,
        gamma=gamma,
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
            evaluation_strategy="epoch",
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
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        # callbacks=[],
        run_initial_eval=False,  # 启用训练前评估
        log_dir=os.path.join(user_home, "DecodingMatters", "log", category_str)
        # log_dir="/home/l/linzijie/DecodingMatters/log/" + category_str + "/"  # 指定日志目录
    )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print('It is going to save the model')
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
