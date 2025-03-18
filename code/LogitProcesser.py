from transformers.generation import LogitsProcessor
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import math
import numpy as np
import torch

from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""



class PrefixConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed


def get_hash(x):
    x = [str(_) for _ in x]
    return '-'.join(x)

class CFEnhancedLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        tokenizer,
        model,
        cf_logits,
        cf_dict,
        guidance_scale: float,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self._num_beams = num_beams
        self.guidance_scale = guidance_scale
        self.tokenizer = tokenizer
        self.cf_logits = cf_logits
        self.cf_dict = cf_dict
        self.count=0


    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        mask = torch.full_like(scores, -1000000)
        cf_score = torch.full_like(scores, 1.0)
        for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
            for beam_id, sent in enumerate(beam_sent):
                if self.count == 0:
                    hash_key = sent[-4:]
                else:
                    hash_key=sent[-self.count:]
                hash_key = hash_key.tolist()
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)

                if len(prefix_allowed_tokens) == 0:
                    continue
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

                temp = []
                if self.cf_logits is not None:
                    # print(self.cf_logits)
                    for allow_token in prefix_allowed_tokens:
                        if self.count == 0:
                            cf_key = [allow_token]
                        else:
                            cf_key = hash_key + [allow_token]
                        if get_hash(cf_key) in self.cf_dict:
                            hash_value = self.cf_dict[get_hash(cf_key)]
                        else:
                            continue

                        sublogits = self.cf_logits[hash_value]
                        temp.append(sublogits.sum() + 1e-20) # max or sum
                    temp = torch.tensor(temp)
                    temp = temp / temp.sum()
                    cf_score[batch_id * self._num_beams + beam_id].scatter_(dim = -1, index=torch.tensor(prefix_allowed_tokens).to(cf_score.device), src=temp.to(cf_score.device))
        cf_score = torch.log(cf_score)
        cf_score = cf_score + mask
        self.count += 1

        if self.guidance_scale == 1:
            scores = scores + mask
            return scores

        scores = scores + mask
        out = self.guidance_scale * (scores - cf_score) + cf_score

        return out


# class TrieLogitsProcessor(LogitsProcessor):
#
#     def __init__(
#             self,
#             tokenizer,
#             trie,
#             num_beams: int,
#             unconditional_ids: Optional[torch.LongTensor] = None,
#             unconditional_attention_mask: Optional[torch.LongTensor] = None,
#             use_cache: Optional[bool] = True,
#             alpha: float = 1.0  # 乘法因子
#     ):
#         self.unconditional_context = {
#             "input_ids": unconditional_ids,
#             "attention_mask": unconditional_attention_mask,
#             "use_cache": use_cache,
#             "past_key_values": None,
#             "first_pass": True,
#         }
#         self._num_beams = num_beams
#         self.tokenizer = tokenizer
#         self.count = 0
#
#         # Trie 相关
#         self.trie = trie
#         self.alpha = torch.tensor(float(alpha))
#         self.batch_size = None
#
#     @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         batch_size, vocab_size = scores.shape
#
#         # 初始化 batch_size（只需要一次）
#         if self.batch_size is None:
#             self.batch_size = batch_size // self._num_beams
#
#         # 跟踪活跃的 beam
#         active_beams = input_ids.ne(self.tokenizer.pad_token_id).any(dim=-1)
#
#         # **初始化 mask 为 inf**，表示默认无效
#         mask = torch.full_like(scores, float('inf'))
#
#         # 为了确保 scores 始终小于 0
#         scores = scores - 1e-7  # 给 scores 减去一个非常小的数
#
#         # 遍历 batch 和 beam
#         for batch_id in range(self.batch_size):
#             for beam_id in range(self._num_beams):
#                 global_idx = batch_id * self._num_beams + beam_id
#
#                 # 跳过已经完成的 beam
#                 if not active_beams[global_idx]:
#                     continue
#
#                 # 获取当前序列的 tokens
#                 current_tokens = input_ids[global_idx].tolist()
#
#                 # 获取当前序列的最后 n 个 tokens
#                 if self.count == 0:
#                     # 第一个 token 的特殊处理
#                     current_score, next_token_scores = self.trie.get_next_token_scores([])
#                 else:
#                     last_tokens = current_tokens[-self.count:]
#
#                     # 获取当前序列的得分和可能的下一个 token 的得分
#                     current_score, next_token_scores = self.trie.get_next_token_scores(last_tokens)
#
#                 # 如果当前序列不可能，将整个 beam 置为无效
#                 if current_score == float('-inf'):
#                     scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
#                     active_beams[global_idx] = False
#                     continue
#
#                 score_diffs = {token: next_score - current_score for token, next_score in next_token_scores.items()}
#
#                 if score_diffs:
#                     diffs_tensor = torch.tensor(list(score_diffs.values()), dtype=torch.float, device=scores.device)
#                     # penalties = (diffs_tensor) * self.alpha
#                     # mask[global_idx, list(score_diffs.keys())] = torch.exp(-penalties * math.log(1.2 + 0.3 * self.count))
#                     min_diff, max_diff = diffs_tensor.min(), diffs_tensor.max()
#
#                     if max_diff > min_diff:
#                         scale = torch.reciprocal(max_diff - min_diff + 1e-7)  # 计算 1 / (max_diff - min_diff)
#                         penalties = (diffs_tensor - min_diff) * scale
#                         mask[global_idx, list(score_diffs.keys())] = torch.exp(-penalties * torch.log(self.alpha))
#                     else:
#                         mask[global_idx, list(score_diffs.keys())] = 1.0
#
#         # **通过乘法调整 scores**
#         scores = scores * mask
#
#         self.count += 1
#         return scores

# class TrieLogitsProcessor(LogitsProcessor):
#     def __init__(
#             self,
#             tokenizer,
#             trie,
#             num_beams: int,
#             unconditional_ids: Optional[torch.LongTensor] = None,
#             unconditional_attention_mask: Optional[torch.LongTensor] = None,
#             use_cache: Optional[bool] = True,
#             alpha: float = 0.0  # 乘法因子
#     ):
#         self.unconditional_context = {
#             "input_ids": unconditional_ids,
#             "attention_mask": unconditional_attention_mask,
#             "use_cache": use_cache,
#             "past_key_values": None,
#             "first_pass": True,
#         }
#         self._num_beams = num_beams
#         self.tokenizer = tokenizer
#         self.count = 0
#
#         # Trie 相关
#         self.trie = trie
#         self.alpha = torch.tensor(float(alpha))
#         self.batch_size = None
#
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
#         batch_size, vocab_size = scores.shape
#
#         # 初始化 batch_size（只需要一次）
#         if self.batch_size is None:
#             self.batch_size = batch_size // self._num_beams
#
#         # 跟踪活跃的 beam
#         active_beams = input_ids.ne(self.tokenizer.pad_token_id).any(dim=-1)
#
#         # 遍历 batch 和 beam
#         for batch_id in range(self.batch_size):
#             for beam_id in range(self._num_beams):
#                 global_idx = batch_id * self._num_beams + beam_id
#
#                 # 跳过已经完成的 beam
#                 if not active_beams[global_idx]:
#                     continue
#
#                 # 获取当前序列的 tokens
#                 current_tokens = input_ids[global_idx].tolist()
#                 if self.count == 0:
#                     # 第一个 token 特殊处理
#                     current_score, next_token_scores = self.trie.get_next_token_scores([])
#                 else:
#                     last_tokens = current_tokens[-self.count:]
#                     current_score, next_token_scores = self.trie.get_next_token_scores(last_tokens)
#
#                 # 如果当前序列不可能，将整个 beam 置为无效
#                 if current_score == float('-inf'):
#                     scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
#                     active_beams[global_idx] = False
#                     continue
#
#                 # 计算 类似信息增益的差值
#                 score_diffs = {token: next_score - current_score for token, next_score in next_token_scores.items()}
#
#                 if score_diffs:
#                     diffs_tensor = torch.tensor(list(score_diffs.values()), dtype=torch.float, device=scores.device)
#                     min_diff, max_diff = diffs_tensor.min(), diffs_tensor.max()
#
#                     if max_diff > min_diff:
#                         scale = torch.reciprocal(max_diff - min_diff + 1e-7)  # 计算 1 / (max_diff - min_diff)
#                         penalties = (diffs_tensor - min_diff) * scale
#                     else:
#                         penalties = torch.zeros_like(diffs_tensor)
#
#                     scores[global_idx, list(score_diffs.keys())] *= (1 - self.alpha * penalties)
#
#         self.count += 1
#         return scores

class TrieLogitsProcessor(LogitsProcessor):

    def __init__(
            self,
            tokenizer,
            trie,
            num_beams: int,
            unconditional_ids: Optional[torch.LongTensor] = None,
            unconditional_attention_mask: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = True,
            alpha: float = 1.0  # 乘法因子
    ):
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self._num_beams = num_beams
        self.tokenizer = tokenizer
        self.count = 0

        # Trie 相关
        self.trie = trie
        self.alpha = torch.tensor(float(alpha))
        self.batch_size = None

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape

        # 初始化 batch_size（只需要一次）
        if self.batch_size is None:
            self.batch_size = batch_size // self._num_beams

        # 跟踪活跃的 beam
        active_beams = input_ids.ne(self.tokenizer.pad_token_id).any(dim=-1)

        # **初始化 mask 为 inf**，表示默认无效
        mask = torch.full_like(scores, float('inf'))

        # 为了确保 scores 始终小于 0
        scores = scores - 1e-7  # 给 scores 减去一个非常小的数

        # 遍历 batch 和 beam
        for batch_id in range(self.batch_size):
            for beam_id in range(self._num_beams):
                global_idx = batch_id * self._num_beams + beam_id

                # 跳过已经完成的 beam
                if not active_beams[global_idx]:
                    continue

                # 获取当前序列的 tokens
                current_tokens = input_ids[global_idx].tolist()

                # 获取当前序列的最后 n 个 tokens
                if self.count == 0:
                    # 第一个 token 的特殊处理
                    current_score, next_token_scores = self.trie.get_next_token_scores([])
                else:
                    last_tokens = current_tokens[-self.count:]

                    # 获取当前序列的得分和可能的下一个 token 的得分
                    current_score, next_token_scores = self.trie.get_next_token_scores(last_tokens)

                # 如果当前序列不可能，将整个 beam 置为无效
                if current_score == float('-inf'):
                    scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
                    active_beams[global_idx] = False
                    continue

                score_diffs = {token: next_score - current_score for token, next_score in next_token_scores.items()}

                if score_diffs:
                    diffs_tensor = torch.tensor(list(score_diffs.values()), dtype=torch.float, device=scores.device)
                    min_diff, max_diff = diffs_tensor.min(), diffs_tensor.max()

                    if max_diff > min_diff:
                        scale = torch.reciprocal(max_diff - min_diff + 1e-7)  # 计算 1 / (max_diff - min_diff)
                        IG_scaled = (diffs_tensor - min_diff) * scale
                        mask[global_idx, list(score_diffs.keys())] = 1.0 - self.alpha * IG_scaled
                    else:
                        mask[global_idx, list(score_diffs.keys())] = 1.0

        # **通过乘法调整 scores**
        scores = scores * mask

        self.count += 1
        return scores

    # class TrieLogitsProcessor(LogitsProcessor):
    #
    #     def __init__(
    #             self,
    #             tokenizer,
    #             trie,
    #             # prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
    #             num_beams: int,
    #             unconditional_ids: Optional[torch.LongTensor] = None,
    #             unconditional_attention_mask: Optional[torch.LongTensor] = None,
    #             use_cache: Optional[bool] = True,
    #             alpha = 0.0
    #     ):
    #         # self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
    #         self.unconditional_context = {
    #             "input_ids": unconditional_ids,
    #             "attention_mask": unconditional_attention_mask,
    #             "use_cache": use_cache,
    #             "past_key_values": None,
    #             "first_pass": True,
    #         }
    #         self._num_beams = num_beams
    #         self.tokenizer = tokenizer
    #         self.count = 0
    #
    #         # 初始trie相关
    #         self.trie = trie
    #         self.alpha = alpha
    #         self.batch_size = None
    #
    #     @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    #     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    #         batch_size, vocab_size = scores.shape
    #
    #         # 初始化batch_size(只需要一次)
    #         if self.batch_size is None:
    #             self.batch_size = batch_size // self._num_beams
    #
    #         # 跟踪活跃的beam
    #         active_beams = input_ids.ne(self.tokenizer.pad_token_id).any(dim=-1)
    #
    #         # 创建mask矩阵
    #         mask = torch.full_like(scores, float('-inf'))
    #
    #         # 对每个batch和beam进行处理
    #         for batch_id in range(self.batch_size):
    #             for beam_id in range(self._num_beams):
    #                 global_idx = batch_id * self._num_beams + beam_id
    #
    #                 # 跳过已经完成的beam
    #                 if not active_beams[global_idx]:
    #                     continue
    #
    #                 # 获取当前序列的tokens
    #                 current_tokens = input_ids[global_idx].tolist()
    #
    #                 # 获取当前序列的最后n个tokens
    #                 if self.count == 0:
    #                     # 第一个token的特殊处理
    #                     current_score, next_token_scores = self.trie.get_next_token_scores([])
    #                 else:
    #                     last_tokens = current_tokens[-self.count:]
    #
    #                     # 获取当前序列的得分和可能的下一个token的得分
    #                     current_score, next_token_scores = self.trie.get_next_token_scores(last_tokens)
    #
    #                 # 如果当前序列不可能，将整个beam置为无效
    #                 if current_score == float('-inf'):
    #                     scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
    #                     active_beams[global_idx] = False
    #                     continue
    #
    #                 score_diffs = {token: next_score - current_score for token, next_score in next_token_scores.items()}
    #
    #                 if score_diffs:
    #                     min_diff, max_diff = min(score_diffs.values()), max(score_diffs.values())
    #
    #                     if max_diff > min_diff:
    #                         scale = 1.0 / (max_diff - min_diff + 1e-5)  # 预计算归一化分母，避免重复除法
    #                         for token, diff in score_diffs.items():
    #                             mask[global_idx, token] = self.alpha * (diff - min_diff) * scale - self.alpha
    #                             # mask[global_idx, token] = - self.alpha
    #                     else:
    #                         for token in score_diffs:
    #                             mask[global_idx, token] = -self.alpha  # 所有值相同，归一化为 0
    #
    #                 else:
    #                     continue
    #
    #         # 应用mask
    #         scores = scores + mask
    #         self.count += 1
    #         return scores

    # # 为每个batch收集所有score_diffs
    # batch_score_diffs = [[] for _ in range(self.batch_size)]
    # token_maps = [{} for _ in range(self.batch_size * self._num_beams)]
    #
    # # 第一步：收集所有beam的score_diffs
    # for batch_id in range(self.batch_size):
    #     for beam_id in range(self._num_beams):
    #         global_idx = batch_id * self._num_beams + beam_id
    #
    #         if not active_beams[global_idx]:
    #             continue
    #
    #         current_tokens = input_ids[global_idx].tolist()
    #
    #         # 获取当前序列的最后n个tokens
    #         if self.count == 0:
    #             current_score, next_token_scores = self.trie.get_next_token_scores([])
    #         else:
    #             last_tokens = current_tokens[-self.count:]
    #             current_score, next_token_scores = self.trie.get_next_token_scores(last_tokens)
    #
    #         # 处理无效序列
    #         if current_score == float('-inf'):
    #             scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
    #             active_beams[global_idx] = False
    #             continue
    #
    #         score_diffs = {token: next_score - current_score
    #                        for token, next_score in next_token_scores.items()}
    #
    #         if score_diffs:
    #             # 存储该beam的score_diffs和对应的token映射
    #             batch_score_diffs[batch_id].extend(score_diffs.values())
    #             token_maps[global_idx] = score_diffs
    #
    # # 第二步：对每个batch进行统一归一化
    # for batch_id in range(self.batch_size):
    #     if not batch_score_diffs[batch_id]:  # 跳过没有有效score_diffs的batch
    #         continue
    #
    #     # 计算当前batch的最大最小值
    #     batch_min = min(batch_score_diffs[batch_id])
    #     batch_max = max(batch_score_diffs[batch_id])
    #
    #     # 对batch内的每个beam应用归一化
    #     for beam_id in range(self._num_beams):
    #         global_idx = batch_id * self._num_beams + beam_id
    #
    #         if not active_beams[global_idx] or not token_maps[global_idx]:
    #             continue
    #
    #         if batch_max > batch_min:
    #             scale = 1.0 / (batch_max - batch_min + 1e-5)
    #             for token, diff in token_maps[global_idx].items():
    #                 mask[global_idx, token] = self.alpha * (diff - batch_min) * scale - self.alpha
    #         else:
    #             for token in token_maps[global_idx]:
    #                 mask[global_idx, token] = -self.alpha
    #
    # # 应用mask
    # scores = scores + mask
    # self.count += 1
    # return scores

    # def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    #
    #     batch_size, vocab_size = scores.shape
    # 确定 batch 大小（只需初始化一次）
    # if self.batch_size is None:
    #     self.batch_size = batch_size // self._num_beams
    #
    # # 保持 mask 操作不变
    # mask = torch.full_like(scores, -math.inf)
    # for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
    #     for beam_id, sent in enumerate(beam_sent):
    #         if self.count == 0:
    #             hash_key = sent[-4:]
    #         else:
    #             hash_key = sent[-self.count:]
    #         hash_key = hash_key.tolist()
    #         prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, hash_key)
    #
    #         if len(prefix_allowed_tokens) == 0:
    #             continue
    #         mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0.0
    #
    # scores = scores + mask

    # batch_size, vocab_size = scores.shape
    #
    # # 初始化batch_size(只需要一次)
    # if self.batch_size is None:
    #     self.batch_size = batch_size // self._num_beams
    #
    # # 跟踪活跃的beam
    # active_beams = input_ids.ne(self.tokenizer.pad_token_id).any(dim=-1)
    # # 只处理活跃的 beam
    #
    # for batch_id in range(self.batch_size):
    #     for beam_id in range(self._num_beams):
    #         global_idx = batch_id * self._num_beams + beam_id
    #
    #         # 跳过已结束的 beam
    #         if not active_beams[global_idx]:
    #             continue
    #
    #         if self.count == 0:
    #             # 获取匹配的 top-k tokens
    #             top_k_scores, top_k_indices = torch.topk(scores[global_idx], min(self.top_k, vocab_size))
    #             # top_k_tokens = [self.tokenizer.decode(idx.item(), skip_special_tokens=True) for idx in
    #             #                 top_k_indices]
    #
    #             # 计算 top-k 序列的匹配分数
    #             # next_sequence_scores = torch.tensor(
    #             #     [self.trie.get_trie_score(generated_text + token) for token in top_k_tokens],
    #             #     device=scores.device,
    #             #     dtype=scores.dtype
    #             # )
    #             next_sequence_scores = torch.tensor([
    #                 self.trie.get_trie_score_by_tokens([idx.item()])
    #                 for idx in top_k_indices
    #             ], device=scores.device, dtype=scores.dtype)
    #
    #             updated_scores = top_k_scores + self.alpha * (next_sequence_scores)
    #
    #         else:
    #             generated_tokens = input_ids[global_idx].tolist()
    #             generated_tokens = generated_tokens[-self.count:]
    #             # if batch_id == 2:
    #             #     generated_text = self.tokenizer.decode(generated_tokens)
    #             #     generated_sequence_scores = torch.tensor(self.trie.get_trie_score(generated_text),
    #             #                                              device=scores.device,
    #             #                                              dtype=scores.dtype)
    #             #     print(self.count)
    #             #     print(generated_text)
    #             #     print(generated_sequence_scores)
    #             generated_sequence_scores = torch.tensor(
    #                 self.trie.get_trie_score_by_tokens(generated_tokens),
    #                 device=scores.device,
    #                 dtype=scores.dtype
    #             )
    #
    #             if generated_sequence_scores < -10000.0:
    #                 scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
    #                 active_beams[global_idx] = False
    #                 continue
    #
    #             # 获取匹配的 top-k tokens
    #             top_k_scores, top_k_indices = torch.topk(scores[global_idx], min(self.top_k, vocab_size))
    #             # top_k_tokens = [self.tokenizer.decode(idx.item(), skip_special_tokens=True) for idx in top_k_indices]
    #
    #
    #             # # 计算 top-k 序列的匹配分数
    #             # next_sequence_scores = torch.tensor(
    #             #     [self.trie.get_trie_score(generated_text + token) for token in top_k_tokens],
    #             #     device=scores.device,
    #             #     dtype=scores.dtype
    #             # )
    #             next_sequence_scores = torch.tensor([
    #                 self.trie.get_trie_score_by_tokens(generated_tokens + [idx.item()])
    #                 for idx in top_k_indices
    #             ], device=scores.device, dtype=scores.dtype)
    #             updated_scores = top_k_scores + self.alpha * (next_sequence_scores - generated_sequence_scores)
    #
    #         scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
    #         scores[global_idx].scatter_(0, top_k_indices, updated_scores)
    #
    # self.count += 1
    #
    # return scores
