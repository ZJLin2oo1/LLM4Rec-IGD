import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
from datasets import Dataset as HFDataset
class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id


    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

    
class IGDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False, trie=None):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.K = K
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()  
    def __len__(self):
        return len(self.data)


    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response: 
{data_point["output"]}
"""
    
    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response: 
{data_point["output"]}"""


    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        L = len(row['history_item_title']) 
        history = ""

        for i in range(L):
            if i == 0:
                history += row['history_item_title'][i]
            else:
                history += ", " + row['history_item_title'][i]
        target_item = str(row['item_title'])  # 直接保留原始文本

        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]
        return {"input": f"The user has palyed the following {self.category}s before: {history}",
                "output": target_item + '\n',
                "dedup": target_item_id == last_history_item_id}
    
    def pre(self, idx):
        instruction =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[0]}
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)



        prompt = self.generate_prompt(history)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
        history["input"] = ""

        attention_mask = [1] * len(tokens)


        if self.test:
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,

                # "select_index": select_index,
            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        tokens = tokens + golden_tokens
        attention_mask = [1] * len(tokens)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]

        if len(tokens) >= self.max_len:
            print(len(tokens))

        return {
            "input_ids": tokens[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:],
            "labels": labels,
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data)), mininterval=10):
            inputs.append(self.pre(i))

        self.inputs = inputs
    
    def get_all(self):
        temp = []
        for i in range(len(self.data)):
            temp.append(self.get_history(self.data.iloc[i]))
        return temp
    
    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


class CFTDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4,
                 dedup=False, trie=None):
        self.data = pd.read_csv(train_file)
        random.seed(seed)

        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.have_print=False
        self.category = category
        self.K = K
        self.dedup = dedup
        self.instructs = [
        f"Given a list of {category} the user recetenly enjoy, please write a new {category} that the user may bought",
        f"Considering the {category} that has recently captured the user's interest, kindly create a compilation of other {category} that the user might have played prior to this.",
        f"Based on the user's current gaming preference, please draft a list of potential {category} they may have experienced beforehand.",
        f"Reflecting on the {category} the user has taken pleasure in recently, we request that you formulate a list of {category} that may have preceded the user's current enjoyment.",
        f"In light of the recent gaming enjoyment expressed by the user, please assemble a list of {category} that could potentially include past titles the user has engaged with.",
        f"Taking into account the {category} that has lately provided enjoyment to the user, please put together an inventory of {category} the user might have explored previously.",
        f"Given the user's newfound enjoyment of a particular {category}, would you kindly generate a roster of other {category} that might resonate with their past gaming experiences?",
        f"In response to the user's recent fondness for a specific {category}, we seek your assistance in listing possible {category} the user may have delighted in earlier.",
        f"With respect to the {category} currently enjoyed by the user, please compile a suggestive list of {category} they may have played in the past.",
        f"Bearing in mind the {category} that the user has recently been enthralled by, please construct a catalog of other {category} that the user potentially partook in beforehand.",
        f"In relation to the user's recent entertainment with a given {category}, it would be appreciated if you could curate a list of {category} that might form part of the user's previous gaming history."
        ]
        self.get_inputs()
        self.pad_token_id = -100  # 在 labels 中用 -100 表示忽略
        self.sep_token_id = 0  # 用 0 作为输入的分隔符

    def get_history(self, row):
        row['history_item_title'] = eval(row['history_item_title'])
        history = ', '.join(row['history_item_title'])
        target_item = str(row['item_title'])

        target_item_id = row["item_id"]
        last_history_item_id = eval(row["history_item_id"])[-1]

        return {
            "input": f"The user has played the following {self.category}s before: {history}",
            "output": target_item + '\n',
            "dedup": target_item_id == last_history_item_id
        }

    # def pre(self, idx):
    #     instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    #
    # ### Instruction:
    # Recommend the next {self.category} for the user based on their play history.
    #
    # """
    #
    #     tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
    #     history = self.get_history(self.data.iloc[idx])
    #
    #     target_item = history['output']
    #
    #     # 有历史记录的 prompt
    #     prompt_with_history = f"{history['input']}\n### Response:"
    #     input_with_history = tokens + self.tokenizer.encode(prompt_with_history, bos=False, eos=False)
    #
    #     # 无历史记录的 prompt
    #     prompt_without_history = f"### Input:\n### Response:"
    #     input_without_history = tokens + self.tokenizer.encode(prompt_without_history, bos=False, eos=False)
    #
    #     # 👉 使用固定值 0 作为分隔符
    #     sep_token = [0]
    #
    #     combined_input = input_with_history + sep_token + input_without_history
    #     attention_mask = [1] * len(combined_input)
    #
    #     # 生成标签
    #     golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
    #     input_prompt_len = len(combined_input)
    #
    #     # 👉 labels 用 -100 标记忽略部分
    #     labels = [-100] * input_prompt_len + golden_tokens
    #     if len(labels) > self.max_len:
    #         labels = labels[-self.max_len:]
    #         attention_mask = attention_mask[-self.max_len:]
    #         combined_input = combined_input[-self.max_len:]
    #
    #     return {
    #         "input_ids": combined_input,
    #         "attention_mask": attention_mask,
    #         "labels": labels
    #     }
    def generate_example_prompt(self, data_point):
        return f"""### Example {data_point["idx"]}:
{data_point["input"]} 

### Response: 
{data_point["output"]}
"""

    def generate_prompt(self, data_point):
        return f"""### User Input: 
{data_point["input"]}

### Response: 
{data_point["output"]}"""
    def pre(self, idx):
        instruction = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{self.instructs[0]}
"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)

        history = self.get_history(self.data.iloc[idx])
        target_item = history['output']
        history['output'] = ''
        negative_prompt_ids = copy.deepcopy(tokens)

        prompt = self.generate_prompt(history)
        history['input'] = "Unknow"
        prompt_ref = self.generate_prompt(history)
        tokens_ref = tokens + self.tokenizer.encode(prompt_ref, bos=False, eos=False)
        tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)

        history["input"] = ""

        # attention_mask_ref = [1] * len(tokens_ref)

        if self.test:
            attention_mask = [1] * len(tokens)
            return {
                "input_ids": tokens,
                "attention_mask": attention_mask,

                # "select_index": select_index,
            }

        golden_tokens = self.tokenizer.encode(target_item, bos=False, eos=True)
        input_prompt_len = len(tokens)
        input_prompt_ref_len = len(tokens_ref)
        tokens = tokens + golden_tokens
        tokens_ref = tokens_ref + golden_tokens
        attention_mask = [1] * len(tokens)
        attention_mask_ref = [1] * len(tokens_ref)
        labels = [-100] * input_prompt_len + tokens[input_prompt_len:]
        labels_ref = [-100] * input_prompt_ref_len + tokens_ref[input_prompt_ref_len:]

        if input_prompt_len > input_prompt_ref_len:
            padding_len = input_prompt_len - input_prompt_ref_len
            tokens_ref = [self.tokenizer.tokenizer.pad_token_id] * padding_len + tokens_ref
            attention_mask_ref = [0] * padding_len + attention_mask_ref
            labels_ref = [-100] * padding_len + labels_ref
            if not self.have_print:
                print("len cheching:", len(tokens_ref) == len(tokens), len(labels_ref) == len(labels))
                self.have_print = True
        elif input_prompt_len < input_prompt_ref_len:
            padding_len = input_prompt_ref_len - input_prompt_len
            tokens = [self.tokenizer.tokenizer.pad_token_id] * padding_len + tokens
            attention_mask = [0] * padding_len + attention_mask
            labels = [-100] * padding_len + labels
            if not self.have_print:
                print("len cheching:", len(tokens_ref) == len(tokens), len(labels_ref) == len(labels))
                self.have_print = True

        if len(tokens) >= self.max_len:
            print("tokens len beyond the max len, len of tokens:", len(tokens))

        return {
            "input_ids": tokens[-self.max_len:] + tokens_ref[-self.max_len:],
            "attention_mask": attention_mask[-self.max_len:] + attention_mask_ref[-self.max_len:],
            "labels": labels[-self.max_len:] + labels[-self.max_len:],
            "training": True,
        }

    def get_inputs(self):
        inputs = []
        for i in tqdm(range(len(self.data)), mininterval=10):
            inputs.append(self.pre(i))

        self.inputs = inputs

    def __getitem__(self, idx):
        return self.inputs[idx]

    def get_all(self):
        return [self.get_history(self.data.iloc[i]) for i in range(len(self.data))]

    def get_inputs_list(self):
        return self.inputs

