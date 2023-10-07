#!/usr/bin/python3 python

"""
Lists the Experiment baselines and training
"""

import os
from typing import Union, List, Dict, Optional, Any, Tuple
from pathlib import Path
from copy import deepcopy

from miditok import MIDITokenizer
from torch import Tensor, LongTensor, no_grad
from torch.nn import Linear, Embedding, Sequential, Module, Tanh
from transformers import GPT2LMHeadModel, GPT2Config, Seq2SeqTrainingArguments, GenerationConfig
import numpy as np

from classes import Experiment, Baseline, DataConfig, TestingConfig, TokenizationConfig
from dataset import DatasetMIDI, DataCollatorStandard
from models import GPT2LMHeadModelEmbedPooling
from constants import *


class BaselineGen(Baseline):
    def create_dataset(self, files_paths: List[Path], **kwargs):
        return DatasetMIDI(
            files_paths,
            self.data_config.min_seq_len - 1,
            self.data_config.max_seq_len - 1,
            self.tokenizer,
            True,
            kwargs,
        )

    def create_data_collator(self, pad_on_left: bool = False, shift_labels: bool = False) -> DataCollatorStandard:
        return DataCollatorStandard(self.pad_token, self.bos_token, pad_on_left=pad_on_left, shift_labels=shift_labels)

    def create_model(self):
        if not self.tokenizer.is_multi_voc:
            model = GPT2LMHeadModel(self.model_config)
            if self.name[-4:] == "tiny":
                model.transformer.wte = Sequential(
                    Embedding(self.model_config.vocab_size, TINY_DIM, self.pad_token),
                    Linear(TINY_DIM, self.model_config.n_embd),
                    Tanh(),
                )
        else:
            embed_pool_size = [self.embed_pooling_size for _ in range(len(self.tokenizer.len))]
            model = GPT2LMHeadModelEmbedPooling(self.model_config, self.tokenizer.len, embed_pool_size)
        model.generation_config = self.generation_config
        return model


model_config = GPT2Config(
    vocab_size=None,
    n_positions=NB_POS_ENC_PARAMS,
    n_embd=DIM,
    n_layer=NB_LAYERS,
    n_head=NB_HEADS,
    n_inner=D_FFWD,
    resid_pdrop=DROPOUT,
    embd_pdrop=DROPOUT,
    attn_pdrop=DROPOUT,
    use_cache=True,
)
training_config = Seq2SeqTrainingArguments(
    "", False, True, True, False, "steps",
    per_device_train_batch_size=BATCH_SIZE_GEN,
    per_device_eval_batch_size=BATCH_SIZE_TEST_GEN,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_steps=VALID_INTVL,
    learning_rate=LEARNING_RATE_GEN,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRADIENT_CLIP_NORM,
    max_steps=TRAINING_STEPS_GEN,
    lr_scheduler_type=LR_SCHEDULER_GEN,
    warmup_ratio=WARMUP_RATIO_GEN,
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
    gradient_checkpointing=False,
    full_determinism=True,
    use_mps_device=USE_MPS,
    torch_compile=TORCH_COMPILE,
    torch_compile_backend=TORCH_COMPILE_BACKEND,
    torch_compile_mode=TORCH_COMPILE_MODE,
    predict_with_generate=True,
)
data_config = DataConfig(0.01, TEST_SPLIT_GEN, MIN_SEQ_LEN_GEN, MAX_SEQ_LEN_GEN)
test_config = TestingConfig(
    batch_size=BATCH_SIZE_TEST_GEN,
    min_seq_len=MIN_SEQ_LEN_TEST_GEN,
    max_seq_len=MAX_SEQ_LEN_TEST_GEN,
)
generation_config = GenerationConfig(
    max_length=None,
    max_new_tokens=NB_INFERENCES_GEN,
    num_beams=NUM_BEAMS,
    do_sample=True,
    temperature=TEMPERATURE_SAMPLING,
    top_k=TOP_K,
    top_p=TOP_P,
    epsilon_cutoff=EPSILON_CUTOFF,
    eta_cutoff=ETA_CUTOFF,
    use_cache=True,
)

datasets = ["Maestro"]
experiments = []
for dataset in datasets:
    for tokenization in TOKENIZATIONS:
        exp_name = f'gen_{dataset}_{tokenization}'
        baselines = []

        # noBPE and tiny baseline
        """data_conf_, test_conf_, model_conf_, train_conf_, gen_conf_ = \
            map(deepcopy, [data_config, test_config, model_config, training_config, generation_config])
        tok_config = TokenizationConfig(tokenization, deepcopy(TOKENIZER_PARAMS))
        baselines.append(BaselineGen(f"{tokenization}_noBPE_tiny", exp_name, dataset, SEED, tok_config,
                                     model_conf_, train_conf_, data_conf_, test_conf_, gen_conf_))"""
        data_conf_, test_conf_, model_conf_, train_conf_, gen_conf_ = \
            map(deepcopy, [data_config, test_config, model_config, training_config, generation_config])
        tok_config = TokenizationConfig(tokenization, deepcopy(TOKENIZER_PARAMS))
        baselines.append(BaselineGen(f"{tokenization}_noBPE", exp_name, dataset, SEED, tok_config,
                                     model_conf_, train_conf_, data_conf_, test_conf_, gen_conf_))

        # BPE baselines
        for bpe_vocab_size in BPE_VOCAB_SIZES:
            data_conf_, test_conf_, model_conf_, train_conf_, gen_conf_ = \
                map(deepcopy, [data_config, test_config, model_config, training_config, generation_config])
            tok_config = TokenizationConfig(tokenization, deepcopy(TOKENIZER_PARAMS), bpe_vocab_size)
            baselines.append(BaselineGen(f"{tokenization}_bpe{bpe_vocab_size}", exp_name, dataset, SEED, tok_config,
                                         model_conf_, train_conf_, data_conf_, test_conf_, gen_conf_))
        # PVm / PVDm
        for token_combination in ['PVm', 'PVDm']:
            data_conf_, test_conf_, model_conf_, train_conf_, gen_conf_ = \
                map(deepcopy, [data_config, test_config, model_config, training_config, generation_config])
            tok_name = f'{tokenization}{token_combination}'
            tok_config = TokenizationConfig(tok_name, deepcopy(TOKENIZER_PARAMS))
            baselines += [BaselineGen(tok_name, exp_name, dataset, SEED, tok_config, model_conf_, train_conf_,
                                      data_conf_, test_conf_, gen_conf_)]
        # Embedding Pooling
        if tokenization == 'REMI':  # adds CPWord and Octuple for comparison
            for tok in ['CPWord', 'OctupleMono']:
                data_conf_, test_conf_, model_conf_, train_conf_, gen_conf_ = \
                    map(deepcopy, [data_config, test_config, model_config, training_config, generation_config])
                datas = f'{dataset}-short' if tok == 'OctupleMono' else dataset
                tok_config = TokenizationConfig(tok, deepcopy(TOKENIZER_PARAMS))
                baselines += [BaselineGen(tok, exp_name, datas, SEED, tok_config, model_conf_, train_conf_, data_conf_,
                                          test_conf_, gen_conf_, embed_pooling_size=EMBED_POOLING_SIZE)]

        experiments.append(Experiment(exp_name, baselines, dataset))


def save_generation_tokens(prompt: Tensor, continuation: Tensor, tokenizer, out_dir: Path, file_name: Union[int, str]):
    r"""Saves generated tokens, as json and MIDi files.
    :param prompt: original sample (prompt) used for the generation.
    :param continuation: generated sequence, with original_sample at its beginning.
    :param tokenizer: tokenizer object.
    :param out_dir: output directory.
    :param file_name: file name, with no extension (.json and .mid will be added).
    """
    generated = continuation[len(prompt):]
    tokens = [generated, prompt, continuation]  # list compr. as seqs of dif. lengths
    tokens = [seq.tolist() for seq in tokens]
    tokenizer.save_tokens(tokens, out_dir / f'{file_name}.json')
    midi = tokenizer.tokens_to_midi(deepcopy(tokens), time_division=TIME_DIVISION)  # copy as inplace decompose bpe op
    midi.instruments[0].name = f'Continuation of original sample ({len(generated)} tokens)'
    midi.instruments[1].name = f'Original sample ({len(prompt)} tokens)'
    midi.instruments[2].name = f'Original sample and continuation'
    midi.dump(out_dir / f'{file_name}.mid')


class ComputeMetrics:
    def __init__(self, tokenizer: MIDITokenizer, out_dir: Path):
        self.tokenizer = tokenizer
        self.out_dir = out_dir
        self.save_gen_tokens = False

    def __call__(self, eval_pred):
        """Computes metrics.
        Here no accuracy as the results are generated autoregressively with generate.

        :param eval_pred: EvalPrediction containing predictions and labels
        :return: metrics
        """
        predictions, labels = eval_pred

        tse_ = []
        for i, (lab, pred) in enumerate(zip(labels, predictions)):
            # Preprocess tokens
            if pred.ndim == 1:
                lab = lab[lab != -100]
                pred = pred[pred != 0]
            else:
                lab_pad_idx = np.where(lab == -100)[0]
                if len(lab_pad_idx) > 0:
                    lab = lab[lab_pad_idx[-1] + 1:]
                """pred_pad_idx = np.where(pred == -0)[0]
                if len(pred_pad_idx) > 0:
                    pred = pred[:pred_pad_idx[0]]"""
            if self.save_gen_tokens:
                save_generation_tokens(lab, pred, self.tokenizer, self.out_dir, i)
            tse_.append(list(tse(pred.tolist(), self.tokenizer)))

        tse_ = np.array(tse_)
        metric_res = {
            "tse_type": float(np.mean(tse_[:, 0])),
            "tse_time": float(np.mean(tse_[:, 1])),
            "tse_ndup": float(np.mean(tse_[:, 2])),
            "tse_nnon": float(np.mean(tse_[:, 3])),
            "tse_nnof": float(np.mean(tse_[:, 4])),
        }

        return metric_res


if __name__ == '__main__':
    import json

    from miditoolkit import MidiFile, Marker
    from transformers import Seq2SeqTrainer
    from transformers.trainer_utils import set_seed, PredictionOutput, get_last_checkpoint
    from tqdm import tqdm

    from dataset import _add_bos_eos_tokens_to_batch, _pad_batch
    from training import train_model, test_model, split_object, is_training_done, is_testing_done, select_device
    from metrics import tse


    class GenTrainer(Seq2SeqTrainer):
        """
        Subclass modifying pad_on_left attribute of data collator for generation.
        """

        def training_step(self, *args, **kwargs):
            self.data_collator.pad_on_left = False
            return super().training_step(*args, **kwargs)

        def evaluate(self, *args, **kwargs):
            self.data_collator.pad_on_left = True
            return super().evaluate(*args, **kwargs)

        def predict(self, *args, **kwargs) -> PredictionOutput:
            self.data_collator.pad_on_left = True
            self.compute_metrics.save_gen_tokens = True
            return super().predict(*args, **kwargs)

        def prediction_step(
                self,
                model: Module,
                inputs: Dict[str, Union[Tensor, Any]],
                prediction_loss_only: bool,
                ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[float], Optional[Tensor], Optional[Tensor]]:

            if not self.args.predict_with_generate or prediction_loss_only:
                return super().prediction_step(
                    model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
                )

            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)

            # XXX: adapt synced_gpus for fairscale as well
            # Priority (handled in generate):
            # gen_kwargs > model.generation_config > default GenerationConfig()
            gen_kwargs = self._gen_kwargs.copy()
            if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
                gen_kwargs["max_length"] = self.model.config.max_length
            gen_kwargs["num_beams"] = (
                gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
            )

            # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
            # (otherwise, it would continue generating from the padded `decoder_input_ids`)
            if (
                    "labels" in inputs
                    and "decoder_input_ids" in inputs
                    and inputs["labels"].shape == inputs["decoder_input_ids"].shape
            ):
                inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
            generated_tokens = self.model.generate(**inputs, **gen_kwargs)

            # Retrieves GenerationConfig from model.generation_config
            gen_config = self.model.generation_config

            with no_grad():
                if has_labels:
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if self.label_smoother is not None:
                        loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    else:
                        loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                else:
                    loss = None

            if self.args.prediction_loss_only:
                return loss, None, None

            if has_labels:
                labels = inputs["labels"]
            else:
                labels = None

            return loss, generated_tokens, labels


    for exp_ in experiments:

        # Split data here, so that we use the exact same test files for all baselines
        # Doing so allows fair human evaluation of the same conditional / prompted generation
        # We assume they have the same data_config
        set_seed(exp_.baselines[0].seed)
        files_names = [p.relative_to(exp_.baselines[0].tokens_path)
                       for p in exp_.baselines[0].tokens_path.glob('**/*.json')]
        names_train, names_valid, names_test = split_object(files_names, [exp_.baselines[0].data_config.valid_ratio,
                                                                          exp_.baselines[0].data_config.test_ratio])

        for baseline_ in exp_.baselines:
            if is_testing_done(baseline_.run_path):
                continue
            if baseline_.name.endswith("PVDm"):
                baseline_.training_config.do_eval = False  # prevents OOM when resuming training on V100s
                baseline_.training_config.evaluation_strategy = "no"
            if baseline_.tokenization == "CPWord":
                for key in baseline_.tokenizer.tokens_types_graph:
                    baseline_.tokenizer.tokens_types_graph[key].append("Ignore")
                baseline_.tokenizer.tokens_types_graph["Ignore"] = baseline_.tokenizer.tokens_types_graph.keys()
            last_checkpoint = get_last_checkpoint(str(baseline_.run_path)) if baseline_.run_path.exists() else None
            baseline_.training_config.resume_from_checkpoint = last_checkpoint
            model_ = baseline_.create_model()
            (gen_path := baseline_.run_path / "gen").mkdir(parents=True, exist_ok=True)

            # Load data
            if baseline_.tokenization == "OctupleMono":
                stems_train = [path.stem for path in names_train]
                stems_valid = [path.stem for path in names_valid]
                stems_test = [path.stem for path in names_test]
                json_paths = list(baseline_.tokens_path.glob("**/*.json"))
                paths_train, paths_valid, paths_test = [], [], []
                for json_path in json_paths:
                    parts = json_path.stem.split("_")
                    original_name = "_".join(parts[:-1])

                    for stem_list, path_list in [(stems_train, paths_train), (stems_valid, paths_valid),
                                                 (stems_test, paths_test)]:
                        if original_name in stem_list:
                            path_list.append(json_path)
            else:
                paths_train = [baseline_.tokens_path / name for name in names_train]
                paths_valid = [baseline_.tokens_path / name for name in names_valid]
                paths_test = [baseline_.tokens_path / name for name in names_test]
            dataset_train = baseline_.create_dataset(files_paths=paths_train)
            dataset_valid = baseline_.create_dataset(files_paths=paths_valid)
            dataset_test = baseline_.create_dataset(files_paths=paths_test)
            collator = baseline_.create_data_collator()
            # Train model if not already done
            trainer = GenTrainer(
                model=model_,
                args=baseline_.training_config,
                data_collator=collator,
                train_dataset=dataset_train,
                eval_dataset=dataset_valid,
                compute_metrics=ComputeMetrics(baseline_.tokenizer, gen_path),
            )
            if not is_training_done(baseline_.run_path):
                train_model(trainer)
            else:
                kwargs = {}
                if baseline_.tokenizer.is_multi_voc:
                    embed_pool_size = [baseline_.embed_pooling_size for _ in range(len(baseline_.tokenizer.len))]
                    kwargs = {"num_classes": baseline_.tokenizer.len, "embed_sizes": embed_pool_size}
                model_ = model_.from_pretrained(last_checkpoint, **kwargs)
            test_model(trainer, dataset_test=dataset_test)

            # Test manually
            """from torch.utils.data import DataLoader
            collator.pad_on_left = True
            device = select_device(not baseline_.training_config.no_cuda, baseline_.training_config.use_mps_device)
            model_ = model_.to(device)
            dataloader = DataLoader(dataset_test, baseline_.test_config.batch_size, collate_fn=collator)
            all_gen_tokens, labels = [], []
            with no_grad():
                for batch in tqdm(dataloader, desc=f"Testing manually {baseline_.name}"):
                    for key in batch:
                        if isinstance(batch[key], Tensor):
                            batch[key] = batch[key].to(device)
                    generated_tokens = model_.generate(**batch, generation_config=baseline_.generation_config)  # (N,T)
                    labels += [seq.numpy() for seq in batch["labels"].cpu()]
                    all_gen_tokens += [seq.numpy() for seq in generated_tokens.cpu()]
            # Saved when calling ComputeMetrics
            # for seq, lab in zip(all_gen_tokens, labels):
            #    save_generation_tokens(lab, seq, baseline_.tokenizer, self.out_dir, i)
            metrics = ComputeMetrics(baseline_.tokenizer, gen_path)
            metrics.save_gen_tokens = True
            metric_res = metrics((all_gen_tokens, labels))
            trainer.log_metrics("test", metric_res)
            trainer.save_metrics("test", metric_res)"""

        # Generate examples for human evaluation
        # Gather tokens
        gen_dir = exp_.run_path / "gen"
        if (gen_dir / "all_tokens.json").is_file():
            continue
        gen_dir.mkdir(parents=True, exist_ok=True)
        set_seed(exp_.baselines[0].seed)
        device = select_device(not exp_.baselines[0].training_config.no_cuda,
                               exp_.baselines[0].training_config.use_mps_device)
        test_midi_paths = [exp_.data_path_midi / name.with_suffix(".mid") for name in names_test]
        tokens = {baseline_.name: [] for baseline_ in exp_.baselines}
        gen_tokens = {baseline_.name: [] for baseline_ in exp_.baselines}
        checkpoints = {baseline_.name: get_last_checkpoint(str(baseline_.run_path)) for baseline_ in exp_.baselines}
        for midi_path in tqdm(test_midi_paths, desc=f"Loading examples for human evals ({exp_.name})"):
            if "§" in midi_path.name:  # augmented tokens json, midi does exist
                continue
            midi_prompt = MidiFile(midi_path)
            max_tick = NB_BEATS_PROMPT_GEN * midi_prompt.ticks_per_beat
            midi_prompt.instruments[0].notes = [n for n in midi_prompt.instruments[0].notes if n.start <= max_tick]
            if len(midi_prompt.instruments[0].notes) < MIN_NB_NOTES_PROMPT_GEN:
                continue

            # Create batches for each baseline
            for baseline_ in exp_.baselines:
                tokens[baseline_.name].append(
                    {"input_ids": LongTensor(baseline_.tokenizer(midi_prompt)[0].ids)}
                )

        # Generates tokens
        for baseline_ in tqdm(exp_.baselines, desc=f"Generating examples for human evals ({exp_.name})"):
            model_ = baseline_.create_model()
            kwargs = {}
            if baseline_.tokenizer.is_multi_voc:
                embed_pool_size = [baseline_.embed_pooling_size for _ in range(len(baseline_.tokenizer.len))]
                kwargs = {"num_classes": baseline_.tokenizer.len, "embed_sizes": embed_pool_size}
            model_ = model_.from_pretrained(checkpoints[baseline_.name], **kwargs).eval().to(device)

            for i in range(0, len(tokens[baseline_.name]), BATCH_SIZE_TEST_GEN):
                input_ids = tokens[baseline_.name][i: i + BATCH_SIZE_TEST_GEN]
                _add_bos_eos_tokens_to_batch(input_ids, bos_tok=baseline_.bos_token)
                input_ids = _pad_batch(input_ids, baseline_.pad_token, pad_on_left=True).to(device)
                if baseline_.tokenizer.is_multi_voc:
                    attention_mask = (input_ids[..., 0] != baseline_.pad_token).int()
                else:
                    attention_mask = (input_ids != baseline_.pad_token).int()
                generated_tokens = model_.generate(input_ids, baseline_.generation_config,
                                                   attention_mask=attention_mask)  # (N,T)
                gen_tokens[baseline_.name] += generated_tokens.tolist()

        # Saves generated tokens as MIDIs
        max_tick = (NB_BEATS_CONTINUATION_MAX + NB_BEATS_PROMPT_GEN) * TIME_DIVISION
        with open(gen_dir / "all_tokens.json", "w") as outfile:
            json.dump(gen_tokens, outfile)
        for i in range(len(gen_tokens[exp_.baselines[0].name])):
            midi_out = MidiFile(ticks_per_beat=TIME_DIVISION)
            midi_out.markers.append(Marker("Continuation starts ~ here", NB_BEATS_PROMPT_GEN * TIME_DIVISION))
            for baseline_ in exp_.baselines:
                toks = gen_tokens[baseline_.name][i]
                if not baseline_.tokenizer.unique_track:
                    toks = [toks]
                midi_conv = baseline_.tokenizer(toks, time_division=TIME_DIVISION)
                # Discard notes after the max nb beats of continuation
                midi_conv.instruments[0].notes = [n for n in midi_conv.instruments[0].notes if n.start < max_tick]
                midi_out.instruments.append(midi_conv.instruments[0])
            midi_out.dump(gen_dir / f"{i}.mid")
