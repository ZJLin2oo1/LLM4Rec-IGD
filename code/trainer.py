import os
import json
import torch
from datetime import datetime
from transformers import Trainer
import numpy as np


class IGMonitorTrainer(Trainer):
    """
    IGMonitorTrainer monitors token-level Information Gain (IG) values and applies IGD-Tuning
    by reweighting token losses during training.

    Args:
        rf_item_trie: A token trie built from the training set that represents a reference distribution over item tokens.
        run_initial_eval (bool): If True, runs initial evaluation before training starts.
        log_dir (str): Directory to save training logs. If None, logging is disabled.
        beta (float): Reweighting factor for tokens with zero IG.
    """
    def __init__(self, rf_item_trie, run_initial_eval=False, log_dir=None, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rf_item_trie = rf_item_trie
        self.run_initial_eval = run_initial_eval
        self.beta = beta  # Reweighting factor for zero-IG tokens
        self.log_file = None

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.jsonl")
            with open(self.log_file, 'w') as f:
                f.write("")

        if self.run_initial_eval:
            self.evaluate()

    def log_metrics(self, split, metrics, epoch=None):
        """Log metrics to the log file."""
        if not self.log_file:
            return

        log_entry = {
            "split": split,
            "epoch": epoch if epoch is not None else self.state.epoch,
            "metrics": metrics
        }
        with open(self.log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss with IG-based reweighting."""
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        per_token_loss = per_token_loss.view(shift_labels.size())

        non_pad_mask = shift_labels != -100
        ig_weights = torch.ones_like(per_token_loss, dtype=torch.float32)

        all_ig_tensors = []
        for i, seq_labels in enumerate(shift_labels):
            valid_labels = seq_labels[non_pad_mask[i]].tolist()
            if valid_labels:
                ig_list = self.rf_item_trie.get_sequence_ig(valid_labels)
                if ig_list[-1] == float('-inf'):
                    ig_list = [0] * len(valid_labels)
                all_ig_tensors.append(torch.tensor(ig_list, dtype=torch.float32, device=per_token_loss.device))

        if all_ig_tensors:
            all_ig_tensor = torch.cat(all_ig_tensors)
        else:
            all_ig_tensor = torch.tensor([], dtype=torch.float32, device=per_token_loss.device)

        zero_mask = all_ig_tensor == 0
        positive_mask = all_ig_tensor > 0

        ig_weights[non_pad_mask] = torch.where(zero_mask, self.beta, ig_weights[non_pad_mask])
        ig_weights[non_pad_mask] = torch.where(positive_mask, 1.0, ig_weights[non_pad_mask])

        # Log IG-based group losses
        if self.log_file:
            ig_groups = {
                "zero_ig": {"total_loss": 0.0, "count": 0},
                "positive_ig": {"total_loss": 0.0, "count": 0}
            }
            for token_ig, loss_train in zip(all_ig_tensor, per_token_loss[non_pad_mask]):
                if token_ig < 0:
                    continue
                group = "zero_ig" if token_ig == 0 else "positive_ig"
                ig_groups[group]["total_loss"] += loss_train.item()
                ig_groups[group]["count"] += 1

            self.log_metrics("train", {
                "ig_zero_ig_loss": ig_groups["zero_ig"]["total_loss"] / ig_groups["zero_ig"]["count"] if ig_groups["zero_ig"]["count"] > 0 else float('nan'),
                "ig_positive_ig_loss": ig_groups["positive_ig"]["total_loss"] / ig_groups["positive_ig"]["count"] if ig_groups["positive_ig"]["count"] > 0 else float('nan')
            }, epoch=self.state.epoch)

        new_weights = ig_weights[non_pad_mask]
        valid_token_loss = (per_token_loss[non_pad_mask] * new_weights).sum() / new_weights.sum()

        return (valid_token_loss, outputs) if return_outputs else valid_token_loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                        metric_key_prefix="eval"):
        """Override evaluation loop to log IG-based evaluation metrics."""
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        if not self.log_file:
            return output

        ig_groups = {
            "zero_ig": {"total_loss": 0.0, "count": 0},
            "positive_ig": {"total_loss": 0.0, "count": 0}
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
                            group = "zero_ig" if token_ig == 0 else "positive_ig"
                            ig_groups[group]["total_loss"] += loss_val
                            ig_groups[group]["count"] += 1

        for group, stats in ig_groups.items():
            avg_loss = stats["total_loss"] / stats["count"] if stats["count"] > 0 else float('nan')
            output.metrics[f"{metric_key_prefix}_ig_{group}_loss"] = avg_loss
            output.metrics[f"{metric_key_prefix}_ig_{group}_count"] = stats["count"]
            print(f"IG Group: {group}, Avg Loss: {avg_loss:.4f}, Count: {stats['count']}")

        self.log_metrics("eval", output.metrics, epoch=self.state.epoch)

        return output


# === Pos and CFT Trainers === #

from dataclasses import dataclass
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, PaddingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer import _is_peft_model
from transformers.trainer import (MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES)
from torch.nn import CrossEntropyLoss
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, PaddingStrategy


@dataclass
class DataCollatorForSeq2Seq_v2:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if 'training' in features[0].keys():
            features_ori = []
            features_ref = []
            for feature in features:
                fea_ori = {}
                fea_ref = {}
                for k, v in feature.items():
                    if k == "training":
                        continue
                    sp_len = len(v) // 2
                    v_ori, v_ref = v[:sp_len], v[sp_len:]
                    fea_ori[k] = v_ori
                    fea_ref[k] = v_ref
                features_ori.append(fea_ori)
                features_ref.append(fea_ref)
            batch_ori = self.run_one(features_ori, return_tensors)
            batch_ref = self.run_one(features_ref, return_tensors)
            batch = {}
            for k, v1, v2 in zip(batch_ori.keys(), batch_ori.values(), batch_ref.values()):
                batch[k] = torch.cat([v1, v2], dim=-1)
            return batch
        else:
            return self.run_one(features, return_tensors)

    def run_one(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # reconvert list[None] to None if necessary
        # this might occur when we pass {..., "labels": None}
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # we have to pad the labels manually as we cannot rely on `tokenizer.pad` and we need them to be of the same length to return tensors
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                            (max_label_length + self.pad_to_multiple_of - 1)
                            // self.pad_to_multiple_of
                            * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        # reintroduce side effects via tokenizer that return respective datatypes for the `return_tensors` argument
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                import tensorflow as tf

                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch


class DebiasedTrainer(Trainer):
    have_print: bool = False
    def __init__(self, beta, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.beta = beta
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if model.training:
            inputs_ref = {}
            inputs_ori = {}
            for k, v in inputs.items():
                v1, v2 = torch.chunk(v, 2, dim=-1)
                inputs_ori[k] = v1
                inputs_ref[k] = v2
            inputs = inputs_ori

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # if "special_tokens_mask" in inputs:

        outputs = model(**inputs)
        # inputs['attention_mask'] = inputs['attention_mask_ref']
        # print("inputs:", inputs)
        # inputs['attention_mask'] = 1 - ref_attention_mask # here, special token mask is used for generate masks for reference prompt [0: normal sequence 1:special + history]

        logits = outputs['logits']

        if model.training:
            outputs_ref = model(**inputs_ref)
            logits_ref = outputs_ref['logits']
            diff_logits = logits - logits_ref

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
                if model.training:
                    outputs['logits'] = diff_logits
                    loss_diff = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
                if model.training:
                    outputs['logits'] = diff_logits
                    loss_diff = self.label_smoother(outputs, labels)
        else:
            if not self.have_print:
                print("debias training mode: No label smooth...")
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # next, we compute the debiased loss
            if model.training:
                shift_diff_logits = diff_logits[..., :-1, :].contiguous()
                vocay_size = shift_diff_logits.shape[-1]
                shift_diff_logits = shift_diff_logits.view(-1, vocay_size)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = shift_logits.view(-1, vocay_size)

                shift_labels = inputs['labels'][..., 1:].contiguous()

                with torch.no_grad():  # new addd
                    # beta = 1.5
                    # _flag = shift_labels>-100
                    # flag = _flag.float()
                    # flag = flag.to(shift_labels.device)
                    # flag_sum = torch.sum(flag,dim=-1).reshape(-1,1)
                    # _gap = 1/(beta*flag_sum)
                    # _pos = torch.cumsum(flag,dim=-1)
                    # weight  = 1 - _pos * _gap
                    # weight = torch.where(_flag, weight, torch.zeros_like(weight))
                    # weight = weight.view(-1)
                    _flag = shift_labels > -100
                    flag = _flag.float()
                    flag = flag.to(shift_labels.device)
                    flag_sum = torch.sum(flag, dim=-1).reshape(-1, 1)
                    _gap = (1 - self.beta) / (flag_sum - 1)
                    _pos = torch.cumsum(flag, dim=-1)
                    weight = 1 - (_pos - 1) * _gap
                    weight = torch.where(_flag, weight, torch.zeros_like(weight))
                    weight = weight.view(-1)

                shift_labels = shift_labels.view(-1)
                loss_fct = CrossEntropyLoss(reduction='none')  # new add reduction=False
                # loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
                loss_diff = loss_fct(shift_diff_logits, shift_labels)

                tot_weight = weight.sum()
                loss = (loss * weight).sum() / tot_weight
                loss_diff = (loss_diff * weight).sum() / tot_weight
                # loss_diff = (loss_diff * weight).sum()/nums # new add

        if model.training:
            # print("training, loss_diff weight",alpha)
            if not self.have_print:
                print("alpha:", self.alpha, 'beta:', self.beta)
                self.have_print = True
            loss += self.alpha * loss_diff

        return (loss, outputs) if return_outputs else loss