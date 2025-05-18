from typing import Optional, Dict, Tuple
import torch
import numpy as np
from transformers.generation import LogitsProcessor
from transformers.utils import add_start_docstrings

LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input token indices.
        scores (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
            Logits or log-softmax scores for each vocabulary token.

    Returns:
        `torch.FloatTensor` of shape `(batch_size, vocab_size)`:
            Processed scores after applying trie-based filtering and scaling.
"""

class TrieLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        trie,
        num_beams: int,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        alpha: float = 1.0
    ):
        """
        Trie-based LogitsProcessor to scale logits according to Information Gain (IG).

        Args:
            tokenizer: The tokenizer used to identify the padding token.
            trie: A trie structure providing valid token continuations and scores.
            num_beams (int): Number of beams used in beam search.
            unconditional_ids (Optional[torch.LongTensor]): Optional unconditional context.
            unconditional_attention_mask (Optional[torch.LongTensor]): Attention mask for unconditional context.
            use_cache (Optional[bool]): Whether to use model cache.
            alpha (float): Scaling factor for IG-based modification.
        """
        self.tokenizer = tokenizer
        self.trie = trie
        self.alpha = torch.tensor(float(alpha))
        self.num_beams = num_beams
        self.batch_size = None
        self.count = 0

        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape

        if self.batch_size is None:
            self.batch_size = batch_size // self.num_beams

        # Identify which beams are still active
        active_beams = input_ids.ne(self.tokenizer.pad_token_id).any(dim=-1)

        # Mask: initialized with infinity (default: suppress all)
        mask = torch.full_like(scores, float('inf'))

        # Ensure logits remain negative (prevent accidental max)
        scores = scores - 1e-7

        for batch_id in range(self.batch_size):
            for beam_id in range(self.num_beams):
                global_idx = batch_id * self.num_beams + beam_id
                if not active_beams[global_idx]:
                    continue

                token_sequence = input_ids[global_idx].tolist()
                context_tokens = [] if self.count == 0 else token_sequence[-self.count:]

                current_score, next_token_scores = self.trie.get_next_token_scores(context_tokens)

                # Invalid path in trie
                if current_score == float('-inf'):
                    scores[global_idx] = torch.full_like(scores[global_idx], float('-inf'))
                    active_beams[global_idx] = False
                    continue

                # Compute score differences (Information Gain)
                score_diffs: Dict[int, float] = {
                    token: current_score - next_score
                    for token, next_score in next_token_scores.items()
                }

                if score_diffs:
                    tokens = list(score_diffs.keys())
                    diffs_tensor = torch.tensor(
                        list(score_diffs.values()), dtype=torch.float, device=scores.device
                    )
                    min_diff, max_diff = diffs_tensor.min(), diffs_tensor.max()

                    if max_diff > min_diff:
                        scale = 1.0 / (max_diff - min_diff + 1e-7)
                        IG_scaled = (diffs_tensor - min_diff) * scale
                        mask[global_idx, tokens] = 1.0 - self.alpha * IG_scaled
                    else:
                        mask[global_idx, tokens] = 1.0

        scores = scores * mask
        self.count += 1
        return scores


# === LogitProcessors from D3 Repository === #

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

