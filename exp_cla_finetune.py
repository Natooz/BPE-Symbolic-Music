#!/usr/bin/python3 python

"""
Composer classification
Note density
Next sequence prediction

Elapsed time (beat)
"""

import os
from abc import ABC
from pathlib import Path
from copy import deepcopy
from typing import List, Any, Dict
from dataclasses import dataclass

from transformers import BertForSequenceClassification, BertConfig, TrainingArguments
from torch.nn import Linear, Embedding, Sequential, Tanh

from classes import Experiment, Baseline, DataConfig, TestingConfig, TokenizationConfig
from models import BertForSequenceClassificationEmbeddingPooling
from dataset import DatasetClaComposer, DatasetClaEmotion, DataCollatorStandard, DatasetClaMMD
from constants import *


class BaselineClaFinetune(Baseline, ABC):
    def create_data_collator(self) -> DataCollatorStandard:
        return DataCollatorStandard(self.pad_token, self.bos_token, self.eos_token)

    def create_model(self):
        if not self.tokenizer.is_multi_voc:
            model = BertForSequenceClassification(self.model_config)
            if self.name[-4:] == "tiny":
                model.bert.embeddings.word_embeddings = Sequential(
                    Embedding(self.model_config.vocab_size, TINY_DIM, self.pad_token),
                    Linear(TINY_DIM, self.model_config.hidden_size),
                    Tanh(),
                )
        else:
            embed_sizes = [self.embed_pooling_size for _ in range(len(self.tokenizer.len))]
            model = BertForSequenceClassificationEmbeddingPooling(self.model_config, self.tokenizer.len, embed_sizes)
        return model


class BaselineComposer(BaselineClaFinetune):

    def create_dataset(self, files_paths: List[Path], *args, **kwargs):
        return DatasetClaComposer(files_paths, self.data_config.min_seq_len - 2, self.data_config.max_seq_len - 2,
                                  self.model_config.num_labels)


class BaselineEmotion(BaselineClaFinetune):
    def create_dataset(self, files_paths: List[Path], *args, **kwargs):
        return DatasetClaEmotion(files_paths, self.data_config.min_seq_len - 2, self.data_config.max_seq_len - 2)


class BaselineMMDGenre(BaselineClaFinetune):
    def create_dataset(self, files_paths: List[Path], *args, **kwargs):
        return DatasetClaMMD(files_paths, self.data_config.min_seq_len - 2, self.data_config.max_seq_len - 2,
                             self.model_config.num_labels, genres=True)


class BaselineMMDArtist(BaselineClaFinetune):
    def create_dataset(self, files_paths: List[Path], *args, **kwargs):
        return DatasetClaMMD(files_paths, self.data_config.min_seq_len - 2, self.data_config.max_seq_len - 2,
                             self.model_config.num_labels, artists=True)


@dataclass
class FineTuningTask:
    name: str
    dataset: str
    nb_classes: int
    baseline_cls: Any
    special_arguments: Dict[str, Dict[str, Any]] = None


ftts = [
    # FineTuningTask("composer_150", "GiantMIDI", 150, BaselineComposer),
]

model_config = BertConfig(
    vocab_size=None,
    num_labels=2,  # will be overridden / modified when creating baselines
    hidden_size=DIM,
    num_hidden_layers=NB_LAYERS,
    num_attention_heads=NB_HEADS,
    intermediate_size=D_FFWD,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
    max_position_embeddings=NB_POS_ENC_PARAMS,
    type_vocab_size=2,
)
finetune_config = TrainingArguments(
    "", False, True, True, False, "steps",
    per_device_train_batch_size=BATCH_SIZE_CLA_FT,
    per_device_eval_batch_size=BATCH_SIZE_TEST_CLA,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_steps=VALID_INTVL,
    learning_rate=LEARNING_RATE_CLA_FT,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRADIENT_CLIP_NORM,
    max_steps=TRAINING_STEPS_CLA_FT,
    lr_scheduler_type=LR_SCHEDULER_CLA,
    warmup_ratio=WARMUP_RATIO_CLA,
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
data_config = DataConfig(VALID_SPLIT, TEST_SPLIT, MIN_SEQ_LEN_CLA_FT, MAX_SEQ_LEN_CLA_FT)
test_config = TestingConfig(BATCH_SIZE_TEST_CLA, MIN_SEQ_LEN_TEST_CLA, MAX_SEQ_LEN_TEST_CLA)


def create_baseline(ftt: FineTuningTask, baseline_name: str, exp_name: str, tok_config: TokenizationConfig):
    data_conf_, test_conf_, model_conf_, train_conf_ = \
        map(deepcopy, [data_config, test_config, model_config, finetune_config])
    model_conf_.num_labels = ftt.nb_classes
    if ftt.special_arguments is not None:
        for name, conf in [("data", data_conf_), ("test", test_conf_), ("model", model_conf_),
                           ("train", train_conf_)]:
            if name in ftt.special_arguments:
                for attr, val in ftt.special_arguments[name].items():
                    setattr(conf, attr, val)

    baseline = ftt.baseline_cls(baseline_name, exp_name, ftt.dataset, SEED, tok_config, model_conf_, train_conf_,
                                data_conf_, test_conf_)
    return baseline


experiments = []
for ftt_ in ftts:
    model_config.num_labels = ftt_.nb_classes  # overrides depending on ft task
    for tokenization in TOKENIZATIONS:
        exp_name_ = f'cla_{ftt_.name}_{ftt_.dataset}_{tokenization}'
        baselines = []

        # noBPE and tiny baseline
        tok_config_ = TokenizationConfig(tokenization, deepcopy(TOKENIZER_PARAMS))
        # baselines.append(create_baseline(ftt_, f"{tokenization}_noBPE_tiny", exp_name_, tok_config_))
        baselines.append(create_baseline(ftt_, f"{tokenization}_noBPE", exp_name_, tok_config_))

        # BPE baselines
        for bpe_vocab_size in BPE_VOCAB_SIZES:
            tok_config_ = TokenizationConfig(tokenization, deepcopy(TOKENIZER_PARAMS), bpe_vocab_size)
            baselines.append(create_baseline(ftt_, f"{tokenization}_bpe{bpe_vocab_size}", exp_name_, tok_config_))

        # PVm / PVDm
        for token_combination in ['PVm', 'PVDm']:
            tok_name = f'{tokenization}{token_combination}'
            tok_config_ = TokenizationConfig(tok_name, deepcopy(TOKENIZER_PARAMS))
            baselines.append(create_baseline(ftt_, tok_name, exp_name_, tok_config_))

        # Embedding Pooling
        if tokenization == 'REMI':  # adds CPWord and Octuple for comparison
            for tok in ['CPWord', 'OctupleMono']:
                datas = f'{ftt_.dataset}-short' if tok == 'OctupleMono' else ftt_.dataset
                tok_config_ = TokenizationConfig(tok, deepcopy(TOKENIZER_PARAMS))
                baselines.append(create_baseline(ftt_, tok, exp_name_, tok_config_))
                baselines[-1].dataset = datas
                baselines[-1].tokens_path = Path("data", f"{datas}_{baselines[-1].tokenization}")
                baselines[-1].embed_pooling_size = EMBED_POOLING_SIZE

        experiments.append(Experiment(exp_name_, baselines, ftt_.dataset))


# MMD experiments
for tokenization in ["TSDPlus", "REMIPlus"]:
    for label_type, cls, nb_labels in [("genre", BaselineMMDGenre, MMD_NB_GENRES),
                                       ("artist", BaselineMMDArtist, MMD_NB_ARTISTS)]:
        exp_name_ = f"cla_{label_type}_MMD_{tokenization}"
        TOKENIZER_PARAMS_MMD = deepcopy(TOKENIZER_PARAMS)
        if tokenization == "REMIPlus":
            TOKENIZER_PARAMS_MMD["max_bar_embedding"] = None
        tok_config_ = TokenizationConfig(tokenization, TOKENIZER_PARAMS_MMD)
        ftt_ = FineTuningTask(label_type, "MMD", nb_labels, cls)

        # noBPE and tiny baseline
        baselines = [
            # create_baseline(ftt_, f"{tokenization}_noBPE_tiny", exp_name_, tok_config_),
            create_baseline(ftt_, f"{tokenization}_noBPE", exp_name_, tok_config_)
        ]

        # BPE baselines
        for bpe_vocab_size in BPE_VOCAB_SIZES:
            tok_config_ = TokenizationConfig(tokenization, TOKENIZER_PARAMS_MMD, bpe_vocab_size)
            baselines.append(create_baseline(ftt_, f"{tokenization}_bpe{bpe_vocab_size}", exp_name_, tok_config_))

        # PVm / PVDm
        for token_combination in ['PVm', 'PVDm']:
            tok_name = f'{tokenization}{token_combination}'
            tok_config_ = TokenizationConfig(tok_name, TOKENIZER_PARAMS)
            baselines.append(create_baseline(ftt_, tok_name, exp_name_, tok_config_))

        # Embedding Pooling
        if tokenization == "REMIPlus":
            datas = f"MMD-short"
            tok_config_ = TokenizationConfig("Octuple", TOKENIZER_PARAMS)
            baselines.append(create_baseline(ftt_, "Octuple", exp_name_, tok_config_))
            baselines[-1].dataset = datas
            baselines[-1].tokens_path = Path("data", f"{datas}_{baselines[-1].tokenization}")
            baselines[-1].embed_pooling_size = EMBED_POOLING_SIZE

        experiments.append(Experiment(exp_name_, baselines, ftt_.dataset))


metrics_names = [("f1", []), ("accuracy", []), ("roc_auc", ["multiclass"])]


if __name__ == '__main__':
    from functools import partial

    from torch import argmax, softmax, Tensor, unique
    from torch.distributed import get_world_size, get_rank
    from transformers import Trainer
    from transformers.trainer_utils import get_last_checkpoint, set_seed
    from evaluate import load as load_metric, Metric

    from training import train_model, test_model, is_training_done, is_testing_done, preprocess_logits, create_subsets

    def compute_metrics_classification(eval_pred, metrics_: Dict[str, Metric]):
        """Computes metrics for pretraining.
        Must use proprocess_logits function that converts logits to predictions (argmax or sampling).

        :param eval_pred: EvalPrediction containing predictions and labels
        :param metrics_: metrics
        :return: results
        """
        logits, labels = eval_pred
        predictions = argmax(Tensor(logits), dim=-1).numpy()
        classes_present = unique(Tensor(labels)).numpy()
        logits_auc = Tensor(logits).t()[classes_present].t()  # (N,C) --> (C,N) --> (C',N) --> (N,C')
        logits_auc = softmax(logits_auc, dim=-1).numpy()
        # print(f"label shape: {labels.shape}\nlogits shape: {logits.shape}\npred shape: {predictions.shape}")

        acc = metrics_["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())
        f1 = metrics_["f1"].compute(predictions=predictions.flatten(), references=labels.flatten(), average="macro")
        auc = metrics_["roc_auc"].compute(prediction_scores=logits_auc, references=labels.flatten(),
                                          average="micro", multi_class="ovr")

        if metrics_["accuracy"].process_id == 0:
            metric_res = {"accuracy": acc["accuracy"], "f1": f1["f1"], "roc_auc": auc["roc_auc"]}
        else:
            metric_res = {"accuracy": 0, "f1": 0, "roc_auc": 0}
        return metric_res

    for exp_ in experiments:
        # For maximum nb of samples for all baselines
        max_idx = -1 if exp_.name.split("_")[-1].startswith("REMI") else 4
        tokens_paths = list(Path(exp_.baselines[max_idx].tokens_path).glob('**/*.json'))
        data_max = exp_.baselines[max_idx].create_dataset(tokens_paths)
        max_nb_samples = len(data_max)

        for baseline_ in exp_.baselines:
            # Check training is not already done and init
            if is_testing_done(baseline_.run_path):
                continue
            last_checkpoint = get_last_checkpoint(str(baseline_.run_path)) if baseline_.run_path.exists() else None
            baseline_.training_config.resume_from_checkpoint = last_checkpoint
            model_ = baseline_.create_model()

            # create metrics
            try:
                metrics_func = {metric: load_metric(metric, *args, num_process=get_world_size(), process_id=get_rank(),
                                                    experiment_id=f"bpe_{exp_.name}_{baseline_.name}")
                                for metric, args in metrics_names}
            except RuntimeError:
                metrics_func = {metric: load_metric(metric, *args, experiment_id=f"bpe_{exp_.name}_{baseline_.name}")
                                for metric, args in metrics_names}

            # Load data
            set_seed(baseline_.seed)  # set before loading checkpoint
            token_paths = list(Path(baseline_.tokens_path).glob('**/*.json'))
            dataset_ = baseline_.create_dataset(files_paths=token_paths)
            if len(dataset_) > max_nb_samples:
                dataset_.reduce_nb_samples(max_nb_samples)
            dataset_train, dataset_valid, dataset_test = create_subsets(dataset_, [baseline_.data_config.valid_ratio,
                                                                                   baseline_.data_config.test_ratio])
            collator = baseline_.create_data_collator()

            # Load pretrained weights if necessary
            if last_checkpoint is None:  # no finetuning yet, we load weights from pretrained
                pt_path = Path("runs", "cla_pre_trained", f"{exp_.dataset}_{baseline_.name}")
                model_kwargs = {"num_labels": baseline_.model_config.num_labels}
                if baseline_.tokenizer.is_multi_voc:
                    embed_sizes = [baseline_.embed_pooling_size for _ in range(len(baseline_.tokenizer.len))]
                    model_kwargs["num_classes"] = baseline_.tokenizer.len
                    model_kwargs["embed_sizes"] = embed_sizes
                model_ = model_.from_pretrained(get_last_checkpoint(pt_path), **model_kwargs)

            # Fine-tune model and test it
            trainer = Trainer(
                model=model_,
                args=baseline_.training_config,
                data_collator=collator,
                train_dataset=dataset_train,
                eval_dataset=dataset_valid,
                compute_metrics=partial(compute_metrics_classification, metrics_=metrics_func),
                # preprocess_logits_for_metrics=preprocess_logits,
            )
            if not is_training_done(baseline_.run_path):
                train_model(trainer)
            else:
                model_ = model_.from_pretrained(last_checkpoint)
            test_model(trainer, dataset_test)
