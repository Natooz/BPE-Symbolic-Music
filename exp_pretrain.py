#!/usr/bin/python3 python

"""
Pre-train models for other tasks
"""

import os
from pathlib import Path
from copy import deepcopy
from typing import Dict

from torch import argmax
from torch.distributed import get_world_size, get_rank
from torch.nn import Linear, Embedding, Sequential
from transformers import TrainingArguments
from evaluate import load as load_metric, Metric
from numpy import concatenate

from classes import DataConfig
from exp_cla_finetune import experiments
from constants import *


pretrain_config = TrainingArguments(
    "", False, True, True, False, "steps",
    per_device_train_batch_size=BATCH_SIZE_PT,
    per_device_eval_batch_size=BATCH_SIZE_TEST_CLA,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_steps=VALID_INTVL,
    learning_rate=LEARNING_RATE_PT,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRADIENT_CLIP_NORM,
    max_steps=TRAINING_STEPS_PT,
    lr_scheduler_type=LR_SCHEDULER_PT,
    warmup_ratio=WARMUP_RATIO_PT,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=LOG_STEPS_INTVL,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    no_cuda=not USE_CUDA,
    seed=SEED,
    fp16=USE_AMP,
    local_rank=int(os.getenv("LOCAL_RANK", -1)),  # for DDP
    load_best_model_at_end=True,
    label_smoothing_factor=LABEL_SMOOTHING,
    optim="adamw_torch",
    report_to=["tensorboard"],  # logging_dir will be set within Baseline class
    ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
    ddp_bucket_cap_mb=DDP_BUCKET_CAP_MB,
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    full_determinism=True,
    use_mps_device=USE_MPS,
    torch_compile=TORCH_COMPILE,
    torch_compile_backend=TORCH_COMPILE_BACKEND,
    torch_compile_mode=TORCH_COMPILE_MODE,
)
data_config_pt = DataConfig(VALID_SPLIT, 0, MIN_SEQ_LEN_PT, MAX_SEQ_LEN_PT)

metrics_names = ["accuracy", "f1"]


def preprocess_logits(logits, _):
    """Preprocesses the logits before accumulating them during evaluation.
    This allows to significantly reduce the memory usage and make the training tractable.
    """
    if isinstance(logits[0], list):
        preds = (  # [Z (N,T,*)] --> [Z (N,T)]
            [argmax(log, dim=-1) for log in logits[0]],  # MLM
            argmax(logits[1], dim=-1),  # NSP
        )
    else:
        preds = (argmax(logits[0], dim=-1), argmax(logits[1], dim=-1))  # long dtype
    return preds


def compute_metrics_pt(eval_pred, metrics_: Dict[str, Metric]):
    """Computes metrics for pretraining.
    Must use proprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :param metrics_: metrics
    :return: results
    """
    (predictions_mlm, predictions_nsp), (labels_mlm, labels_nsp) = eval_pred

    # preprocess tensors
    if isinstance(predictions_mlm, list):
        flat = [seq.flatten() for seq in predictions_mlm]
        predictions_mlm = concatenate(flat)
        labels_mlm = labels_mlm.flatten()
    pad_mask = labels_mlm != -100
    labels_mlm, predictions_mlm = labels_mlm[pad_mask], predictions_mlm[pad_mask]
    predictions_nsp, labels_nsp = predictions_nsp.flatten(), labels_nsp.flatten()

    acc = metrics_["accuracy"].compute(predictions=predictions_mlm, references=labels_mlm)
    f1 = metrics_["f1"].compute(predictions=predictions_nsp, references=labels_nsp, average="micro")

    if metrics_["accuracy"].process_id == 0:
        metric_res = {"accuracy_mlm": acc["accuracy"], "f1_nsp": f1["f1"]}
    else:
        metric_res = {"accuracy_mlm": 0, "f1_nsp": 0}

    return metric_res


if __name__ == '__main__':
    from functools import partial

    from transformers import Trainer, BertForPreTraining
    from transformers.trainer_utils import set_seed, get_last_checkpoint

    from dataset import DatasetMIDI, DataCollatorClaPreTrain
    from models import BertForPreTrainingEmbedPooling
    from training import train_model, split_object, is_training_done

    for exp_ in experiments:
        for baseline_ in exp_.baselines:
            pt_path = Path("runs", "cla_pre_trained", f"{exp_.dataset}_{baseline_.name}")
            if is_training_done(pt_path):
                continue
            # Adjust data and training config for pretraining
            baseline_.data_config = data_config_pt
            baseline_.training_config = deepcopy(pretrain_config)
            baseline_.training_config.output_dir = str(pt_path)
            baseline_.training_config.logging_dir = str(pt_path)
            if baseline_.name.endswith("PVm") or baseline_.name.endswith("PVDm") or \
                    (baseline_.tokenization_config.bpe_vocab_size is not None and
                     baseline_.tokenization_config.bpe_vocab_size >= 20000):
                baseline_.training_config.do_eval = False
                baseline_.training_config.evaluation_strategy = "no"

            # create metrics
            try:
                metrics = {metric: load_metric(metric, num_process=get_world_size(), process_id=get_rank(),
                                               experiment_id=f"bpe_{exp_.name}_{baseline_.name}")
                           for metric in metrics_names}
            except RuntimeError:
                metrics = {metric: load_metric(metric, experiment_id=f"bpe_{exp_.name}_{baseline_.name}")
                           for metric in metrics_names}

            # pre-trained weights are common to all subsequent tasks
            last_checkpoint = get_last_checkpoint(str(baseline_.run_path)) if baseline_.run_path.exists() else None
            baseline_.training_config.resume_from_checkpoint = last_checkpoint
            embed_pool_size = None if not baseline_.tokenizer.is_multi_voc else \
                [baseline_.embed_pooling_size for _ in range(len(baseline_.tokenizer.len))]
            if not baseline_.tokenizer.is_multi_voc:
                model_ = BertForPreTraining(baseline_.model_config)
                if baseline_.name[-4:] == "tiny":
                    model_.bert.embeddings.word_embeddings = Sequential(
                        Embedding(baseline_.model_config.vocab_size, TINY_DIM, baseline_.pad_token),
                        Linear(TINY_DIM, baseline_.model_config.hidden_size)
                    )
            else:
                model_ = BertForPreTrainingEmbedPooling(baseline_.model_config, baseline_.tokenizer.len,
                                                        embed_pool_size)

            # Load data
            set_seed(baseline_.seed)
            token_paths = list(Path(baseline_.tokens_path).glob('**/*.json'))
            paths_train, paths_valid, paths_test = split_object(token_paths, [baseline_.data_config.valid_ratio,
                                                                              baseline_.data_config.test_ratio])
            dataset_train = DatasetMIDI(paths_train,
                                        baseline_.data_config.min_seq_len - 3,  # considering SOS, SEP and EOS tokens
                                        baseline_.data_config.max_seq_len - 3,
                                        baseline_.tokenizer)
            dataset_valid = DatasetMIDI(paths_valid,
                                        baseline_.data_config.min_seq_len - 3,
                                        baseline_.data_config.max_seq_len - 3,
                                        baseline_.tokenizer)
            collator = DataCollatorClaPreTrain(pad_token=baseline_.pad_token,
                                               bos_token=baseline_.bos_token,
                                               eos_token=baseline_.eos_token,
                                               mask_token=baseline_.mask_token,
                                               sep_token=baseline_.sep_token,
                                               vocab_size=baseline_.tokenizer.len,
                                               special_tokens=baseline_.special_tokens,
                                               mlm_probability=MASK_RATIO_CLA_PT)

            # Pre-train the model
            trainer = Trainer(
                model=model_,
                args=baseline_.training_config,
                data_collator=collator,
                train_dataset=dataset_train,
                eval_dataset=dataset_valid,
                compute_metrics=partial(compute_metrics_pt, metrics_=metrics),
                preprocess_logits_for_metrics=preprocess_logits,
            )
            train_model(trainer)
