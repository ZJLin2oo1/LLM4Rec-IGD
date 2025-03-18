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
from datasets import Dataset as HFDataset
from my_dataset import IGDataset, CFTDataset
from torch.optim.lr_scheduler import LambdaLR
"""
Unused imports:`
import torch.nn as nn
import bitsandbytes as bnb
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer import _is_peft_model
from transformers.trainer import (MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES)
from torch.nn import CrossEntropyLoss

from dataclasses import dataclass
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, PaddingStrategy
import os
from datetime import datetime

import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
user_home = os.path.expanduser("~")
print(user_home)

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
                    # print("v_ori:",v_ori)
                    # print("v_ref:",v_ref)
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


class DebiasedTrainer(transformers.Trainer):
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
        micro_batch_size: int = 8,
        num_epochs: int = 3,
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
        version: str = "base",
        beta: float = 0.0,
        alpha: float = 0.0

):
    print()
    # print(train_file)
    frequency_scale_dict = {"Books": 675262.0, "Video_Games": 195559.0, "Toys_and_Games": 111806.0,
                            "CDs_and_Vinyl": 134436.0, "Sports_and_Outdoors": 160015.0}
    category_dict = {"Office_Products": "office products", "Books": "books", "steam": "games",
                     "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games",
                     "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors",
                     "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie"}
    frequency_scale = frequency_scale_dict[category]
    print(category)
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
        # device_map=device_map,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    train_data = CFTDataset(train_file=train_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category, K = K)
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

    trainer = DebiasedTrainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        alpha=alpha,
        beta=beta,
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
        data_collator=DataCollatorForSeq2Seq_v2(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        # callbacks=[],
        # log_dir=os.path.join(user_home, "DecodingMatters", "log", category_str)
    )

    model.config.use_cache = False
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print('It is going to save the model')
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
