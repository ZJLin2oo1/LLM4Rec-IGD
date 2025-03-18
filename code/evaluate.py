
import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer
from transformers import AutoModelForCausalLM
from dataset import  D3Dataset
from transformers import  LogitsProcessorList, TemperatureLogitsWarper
from LogitProcesser import CFEnhancedLogitsProcessor, TrieLogitsProcessor
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)


    
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    logits_file: str=None,
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 128,
    K: int = 0,
    seed: int = 0,
    temperature: float=1.0,
    guidance_scale: float=1.0,
    length_penalty: float=1.0
):
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "STEAM": "games" }
    category = category_dict[category]
    print(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, use_flash_attention_2=True if base_model.lower().find("llama") > -1 else False)
    with open(info_file, 'r') as f:
        info = f.readlines()
        info = ["\"" + _.split('\t')[0].strip(' ') + "\"\n" for _ in info]
        item_name = info
        info = [f'''### Response: 
{_}''' for _ in info]
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info]
    
    hash_dict = dict()
    sasrec_dict = dict()
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(4, len(ID)):
            if i == 4:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[4:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
                sasrec_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
            sasrec_dict[hash_number].add(index)
        hash_number = get_hash(ID[4:])
        if hash_number not in sasrec_dict:
            sasrec_dict[hash_number] = set()
        sasrec_dict[hash_number].add(index)

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in sasrec_dict.keys():
        sasrec_dict[key] = list(sasrec_dict[key])
    
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    val_dataset=D3Dataset(train_file=test_data_path, tokenizer=tokenizer,max_len=2560, category=category, test=True,K=K, seed=seed)

    
    if logits_file is not None:
        if not logits_file.endswith(".npy"):
            logits_file = None
    
    if logits_file is not None:
        logits = np.load(logits_file)
        sasrec_logits = torch.tensor(logits).softmax(dim = -1)
        sasrec_logits = sasrec_logits[val_dataset.data['Unnamed: 0'].tolist()]
        
    encodings = [val_dataset.__getitem__(i) for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    model.eval()

    # def evaluate(
    #         encodings,
    #         cf_logits,
    #         temperature=1.0,
    #         num_beams=10,
    #         max_new_tokens=64,
    #         guidance_scale=1.0,
    #         length_penalty=1.0,
    #         **kwargs,
    # ):
    #     maxLen = max([len(_["input_ids"]) for _ in encodings])
    #
    #     padding_encodings = {"input_ids": []}
    #
    #     for _ in encodings:
    #         L = len(_["input_ids"])
    #         padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
    #
    #
    #     generation_config = GenerationConfig(
    #         num_beams=num_beams,
    #         length_penalty=length_penalty,
    #         # top_p=0,
    #         # top_k=10,
    #         num_return_sequences=num_beams,
    #         pad_token_id = model.config.pad_token_id,
    #         eos_token_id = model.config.eos_token_id,
    #         max_new_tokens = max_new_tokens,
    #         **kwargs
    #     )
    #     with torch.no_grad():
    #         ccc = CFEnhancedLogitsProcessor(
    #             guidance_scale=guidance_scale,
    #             cf_logits=cf_logits,
    #             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #             cf_dict=sasrec_dict,
    #             unconditional_ids=None,
    #             model=model,
    #             tokenizer=tokenizer,
    #             num_beams=num_beams
    #         )
    #         logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=temperature), ccc])
    #
    #         generation_output = model.generate(
    #             torch.tensor(padding_encodings["input_ids"]).to(device),
    #             generation_config=generation_config,
    #             return_dict_in_generate=True,
    #             output_scores=True,
    #             logits_processor=logits_processor,
    #         )
    #
    #     s = generation_output.sequences[:, L:]
    #     sequence_scores = [[0 for i in range(len(generation_output.scores))] for _ in range(num_beams)]
    #     for i in range(num_beams):
    #         for j in range(L, len(generation_output.sequences[i])):
    #             beam_index = generation_output.beam_indices[i][j - L]
    #             if beam_index != -1:
    #                 sequence_scores[i][j - L] = generation_output.scores[j - L][beam_index][generation_output.sequences[i][j]].item()
    #     scores = generation_output.sequences_scores.tolist()
    #     output = tokenizer.batch_decode(s, skip_special_tokens=True)
    #     output = [_.split("Response:")[-1] for _ in output]
    #     real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
    #     real_scores = [scores[i * num_beams: (i + 1) * num_beams] for i in range(len(scores) // num_beams)]
    #     return real_outputs, real_scores, sequence_scores

    def evaluate(
            encodings,
            cf_logits,
            temperature=1.0,
            num_beams=10,
            max_new_tokens=64,
            guidance_scale=1.0,
            length_penalty=1.0,
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        padding_encodings = {"input_ids": [], "attention_mask": []}

        for _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])

            # Generate attention_mask
            attention_mask = [0] * (maxLen - L) + [1] * L
            padding_encodings["attention_mask"].append(attention_mask)

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
            ccc = CFEnhancedLogitsProcessor(
                guidance_scale=guidance_scale,
                cf_logits=cf_logits,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                cf_dict=sasrec_dict,
                unconditional_ids=None,
                model=model,
                tokenizer=tokenizer,
                num_beams=num_beams,
            )

            logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=temperature), ccc])

            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(padding_encodings["attention_mask"]).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

        # Only consider the generated part after padding
        s = generation_output.sequences[:, maxLen:]
        scores = generation_output.sequences_scores.tolist()

        # # Calculate sequence scores for generated tokens only
        # sequence_scores = [[[0 for _ in range(s.shape[1])] for _ in range(num_beams)] for _ in range(batch_size)]
        # for batch_idx in range(batch_size):
        #     for beam_idx in range(num_beams):
        #         seq_idx = batch_idx * num_beams + beam_idx
        #         for step_idx in range(s.shape[1]):
        #             beam_index = generation_output.beam_indices[seq_idx][step_idx]
        #             if beam_index != -1:
        #                 sequence_scores[batch_idx][beam_idx][step_idx] = (
        #                     generation_output.scores[step_idx][beam_index][s[seq_idx][step_idx]].item()
        #                 )
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
        if logits_file is not None:
            output, score, seq_score = evaluate(encodings, sasrec_logits[idx].to(device), temperature=temperature, guidance_scale=guidance_scale, length_penalty=length_penalty)
        else:
            output, score, seq_score = evaluate(encodings, cf_logits=None, temperature=temperature, guidance_scale=guidance_scale, length_penalty=length_penalty)
        # outputs = outputs + output
        # scores = scores + score
        # seq_scores.append(seq_score)
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





