
import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer
from transformers import AutoModelForCausalLM
from dataset import  D3Dataset
from transformers import  LogitsProcessorList, TemperatureLogitsWarper
from LogitProcesser import TrieLogitsProcessor

from trie import Trie
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

# def get_hash(x):
#     x = [str(_) for _ in x]
#     return '-'.join(x)

# def modify_items(text):
#     # 给文本开头和结尾添加引号，并在结尾加换行符
#     result = '"' + text.replace(", ", '", "') + '"\n' + '<|endoftext|>'
#     return result

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


def load_titles_to_trie_from_txt(input_file, tokenizer):
    """从 TXT 读取标题并加载到 Trie 中"""
    trie = Trie(tokenizer=tokenizer)

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading titles to Trie", mininterval=10.0):
            parts = line.strip().split('\t')  # 按 `\t` 分割

            title_name = parts[0]  # 取第一部分作为标题
            item_id = int(parts[1])  # 第二部分转换为整数
            modify_title = modify_items(title_name)  # 添加引号、换行符和终结符
            trie.insert(modify_title, 1)  # 插入 Trie

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

def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    reference_item_path = "" ,
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 32,
    K: int = 0,
    seed: int = 0,
    temperature: float=1.0,
    alpha: float=0.0,
    length_penalty: float=0.0
):
    frequency_scale_dict = {"Books": 675262.0, "Video_Games": 195559.0, "Toys_and_Games": 111806.0, "CDs_and_Vinyl": 134436.0, "Sports_and_Outdoors": 160015.0}
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "STEAM": "games" }
    frequency_scale = frequency_scale_dict[category]
    category = category_dict[category]
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, use_flash_attention_2=True if base_model.lower().find("llama") > -1 else False)
    with open(info_file, 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in info]
        item_name = info
        info = [f'''### Response: 
{_}''' for _ in info]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    val_dataset=D3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True,K=K, seed=seed)


    # load trie:
    # rf_item_trie = load_titles_to_trie(reference_item_path, tokenizer=tokenizer)
    # rf_item_trie = load_titles_to_trie_from_txt(reference_item_path, tokenizer=tokenizer)
    rf_item_trie = load_titles_to_trie_from_json(reference_item_path, tokenizer=tokenizer, frequency_scale=frequency_scale)
    encodings = [val_dataset.__getitem__(i) for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.eval()

    def evaluate(
            encodings,
            rf_item_trie,
            temperature=1.0,
            num_beams=10,
            max_new_tokens=72,
            length_penalty=1.0,
            alpha=0.0,
            **kwargs,
    ):

        # 自动 padding
        padded_encodings = tokenizer.pad(
            encodings,
            padding=True,  # 自动 padding
            return_tensors="pt",  # 返回 PyTorch 张量
        )

        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        with torch.no_grad():

            tlp = TrieLogitsProcessor(
                unconditional_ids=None,
                tokenizer=tokenizer,
                num_beams=num_beams,
                trie=rf_item_trie,
                alpha=alpha
            )
            logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=temperature), tlp])

            generation_output = model.generate(
                input_ids=padded_encodings["input_ids"].to(device),
                attention_mask=padded_encodings["attention_mask"].to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
        input_lengths = padded_encodings["input_ids"].shape[1]
        # Only consider the generated part after padding
        s = generation_output.sequences[:, input_lengths:]
        scores = generation_output.sequences_scores.tolist()

        # Calculate sequence scores for generated tokens only
        batch_size = len(encodings)
        sequence_scores = [[[0 for _ in range(s.shape[1])] for _ in range(num_beams)] for _ in range(batch_size)]
        for batch_idx in range(batch_size):
            for beam_idx in range(num_beams):
                seq_idx = batch_idx * num_beams + beam_idx
                for step_idx in range(s.shape[1]):
                    beam_index = generation_output.beam_indices[seq_idx][step_idx]
                    if beam_index != -1:
                        sequence_scores[batch_idx][beam_idx][step_idx] = (
                            generation_output.scores[step_idx][beam_index][s[seq_idx][step_idx]].item()
                        )

        # Decode outputs
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split("Response:")[-1] for _ in output]

        # Restructure outputs and scores by batch
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(batch_size)]
        real_scores = [scores[i * num_beams: (i + 1) * num_beams] for i in range(batch_size)]

        return real_outputs, real_scores, sequence_scores

    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])
    Flg=True
    scores = []
    seq_scores = []
    import random
    for idx, encodings in enumerate(tqdm(new_encodings)):
        output, score, seq_score = evaluate(encodings, rf_item_trie, temperature=temperature, length_penalty=length_penalty, alpha=alpha)

        # outputs = outputs + output
        # scores = scores + score
        # seq_scores.append(seq_score)
        # print(output)
        # print(score)
        # print(seq_score)
        # break
        outputs.extend(output)
        scores.extend(score)
        seq_scores.extend(seq_score)


    # 将生成的结果写入 test_data
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        test["predict_score"] = scores[i]
        test["predict_seq_score"] = seq_scores[i]

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  
    
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)





