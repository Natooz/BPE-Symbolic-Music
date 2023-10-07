#!/usr/bin/python3 python

from pathlib import Path
from time import time
from typing import List, Dict
import logging

from torch import Tensor, LongTensor, no_grad, mean, sum as sum_pt, isin, any as torch_any
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_utils import set_seed, get_last_checkpoint
from tqdm import tqdm
from miditok import TokSequence, MIDITokenizer, OctupleMono
from pandas import DataFrame

from dataset import _add_bos_eos_tokens_to_batch, _pad_batch
from exp_generation import experiments
from training import select_device


class DatasetMidiPaths(Dataset):
    def __init__(self, files_paths: List[Path]):
        self.files_paths = files_paths

    def __getitem__(self, idx) -> Path:
        return self.files_paths[idx]

    def __len__(self):
        return len(self.files_paths)


class DataCollatorReadMidi:
    def __init__(
        self,
        tokenizer: MIDITokenizer,
        max_seq_len: int,
        pad_token: int,
        bos_token: int = None,
        eos_token: int = None,
        pad_on_left: bool = False,
        shift_labels: bool = False,
        labels_pad_idx: int = -100,
        add_bos_eos_to_labels: bool = False,
        inputs_kwarg_name: str = "input_ids",
        labels_kwarg_name: str = "labels",
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_on_left = pad_on_left
        self.shift_labels = shift_labels
        self.labels_pad_idx = labels_pad_idx
        self.add_bos_eos_to_labels = add_bos_eos_to_labels
        self.inputs_kwarg_name = inputs_kwarg_name
        self.labels_kwarg_name = labels_kwarg_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Path]) -> Dict[str, LongTensor]:
        out_batch = {}
        _batch = []

        # Read midis
        for file_path in batch:
            tokens = LongTensor(self.tokenizer(file_path)[0].ids[:self.max_seq_len])
            if isinstance(self.tokenizer, OctupleMono):  # (T,Z)
                if torch_any(tokens[:, 4] >= 64):  # index out of range on bar dimension
                    continue
            _batch.append({self.inputs_kwarg_name: tokens})
        batch = _batch

        # Add BOS and EOS tokens + PAD to inputs
        x = None
        if self.inputs_kwarg_name in batch[0]:
            _add_bos_eos_tokens_to_batch(batch, self.inputs_kwarg_name, bos_tok=self.bos_token, eos_tok=self.eos_token)
            x = _pad_batch(batch, self.pad_token, self.inputs_kwarg_name, self.pad_on_left)

        # Add BOS and EOS tokens + PAD labels
        y = None
        if self.labels_kwarg_name in batch[0]:
            # generation or language modeling
            if isinstance(batch[0][self.labels_kwarg_name], LongTensor):
                _add_bos_eos_tokens_to_batch(batch, self.labels_kwarg_name,
                                             bos_tok=self.bos_token, eos_tok=self.eos_token)
                y = _pad_batch(batch, self.labels_pad_idx, self.labels_kwarg_name, self.pad_on_left)
            # classification
            else:
                y = LongTensor([item[self.labels_kwarg_name] for item in batch])

        # Shift labels
        if self.shift_labels:  # otherwise it's handled in models such as GPT2LMHead
            if x is not None:
                inputs = x
            elif y is not None:
                inputs = y
            else:
                raise ValueError("Either inputs or labels have to be specified by the Dataset.")
            x = inputs[:-1]
            y = inputs[1:]

        # Add inputs / labels to output batch
        if x is not None:
            out_batch[self.inputs_kwarg_name] = x
        if y is not None:
            out_batch[self.labels_kwarg_name] = y

        # Create attention mask (just for padding, causality is handled in models)
        attention_mask = (x != self.pad_token).int()
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[..., 0]  # (N,T,Z) --> (N,T)
        out_batch["attention_mask"] = attention_mask

        return out_batch


if __name__ == "__main__":
    (out_dir := Path("analysis")).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("inference_speed_ete")
    logger.addHandler(logging.FileHandler(out_dir / "inference_speed_ete.log"))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    set_seed(444)
    batch_sizes = [1, 8, 64]
    nb_batches = 10
    indexes = ["No BPE", "BPE 1k", "BPE 5k", "BPE 10k", "BPE 20k", "PVm", "PVDm", "CPWord", "Octuple"]
    metrics = ["Nb beats", "Nb base tokens", "Nb notes"]
    df = DataFrame(columns=[f"{m} - {b} - {tok}" for m in metrics for b in batch_sizes for tok in ["TSD", "REMI"]],
                   index=indexes)

    for exp in experiments:
        midi_files_paths = list(exp.data_path_midi.glob("**/*.mid"))
        for bi, baseline in enumerate(exp.baselines):
            logger.debug(
                f"\n{exp.name} - {baseline.name} - vocab size: {len(baseline.tokenizer)} tokens"
            )

            # Loading model
            device = select_device(
                not baseline.training_config.no_cuda,
                baseline.training_config.use_mps_device,
            )
            last_checkpoint = get_last_checkpoint(str(baseline.run_path))
            model_ = baseline.create_model()
            kwargs = {}
            if baseline.tokenizer.is_multi_voc:
                embed_pool_size = [
                    baseline.embed_pooling_size
                    for _ in range(len(baseline.tokenizer.len))
                ]
                kwargs = {
                    "num_classes": baseline.tokenizer.len,
                    "embed_sizes": embed_pool_size,
                }
            model_ = model_.from_pretrained(last_checkpoint, **kwargs).to(device)
            collator = DataCollatorReadMidi(
                baseline.tokenizer,
                baseline.test_config.max_seq_len,
                baseline.pad_token,
                baseline.bos_token,
                pad_on_left=True
            )

            # Run benchmarks for several batch sizes
            for batch_size in batch_sizes:
                logger.debug(f"Batch size: {batch_size}")
                dataset = DatasetMidiPaths(midi_files_paths[:batch_size * nb_batches])
                dataloader = DataLoader(dataset, batch_size, collate_fn=collator)

                # Generate
                all_gen_tokens, all_gen_midi = [], []
                nb_seq_generated = 0
                with no_grad():
                    t0 = time()
                    for batch_ in tqdm(
                        dataloader, desc=f"Measuring inference speed {baseline.name}"
                    ):
                        batch_size, prompt_len = batch_["input_ids"].shape[:2]
                        for key in batch_:
                            if isinstance(batch_[key], Tensor):
                                batch_[key] = batch_[key].to(device)

                        generated_tokens = model_.generate(
                            **batch_, generation_config=baseline.generation_config
                        )  # (N,T)
                        for seq in generated_tokens.cpu().tolist():
                            seq_ = seq[prompt_len:]
                            all_gen_tokens.append(seq_)
                            all_gen_midi.append(
                                baseline.tokenizer([seq_] if not baseline.tokenizer.unique_track else seq_)
                            )
                        nb_seq_generated += batch_size
                total_time = time() - t0
                mean_time = total_time / nb_seq_generated

                # Decode tokens and compute inference speeds
                nb_beats, nb_base_tokens, nb_notes = [], [], []
                for seq, midi in zip(all_gen_tokens, all_gen_midi):
                    nb_beats.append(midi.max_tick / midi.ticks_per_beat)
                    nb_notes.append(len(midi.instruments[0].notes))

                    # Decode BPE
                    tok_seq = TokSequence(
                        ids=seq, ids_bpe_encoded=baseline.tokenizer.has_bpe
                    )
                    if baseline.tokenizer.has_bpe:
                        baseline.tokenizer.decode_bpe(tok_seq)
                    elif baseline.tokenization.endswith("PVm"):
                        token_ids = LongTensor(baseline.tokenizer.token_ids_of_type("PitchVel"))
                        nb_combined_tok = isin(LongTensor(tok_seq.ids), token_ids)
                        nb_combined_tok = int(sum_pt(nb_combined_tok))
                        tok_seq.ids = tok_seq.ids + ([0] * nb_combined_tok)
                    elif baseline.tokenization.endswith("PVDm"):
                        token_ids = LongTensor(baseline.tokenizer.token_ids_of_type("PitchVelDur"))
                        nb_combined_tok = isin(LongTensor(tok_seq.ids), token_ids)
                        nb_combined_tok = int(sum_pt(nb_combined_tok))
                        tok_seq.ids = tok_seq.ids + ([0] * nb_combined_tok * 2)
                    elif baseline.tokenization == "CPWord":
                        tok_seq.ids = [0] * (len(tok_seq.ids) + len(midi.instruments[0].notes) * 2)

                    nb_base_tokens.append(len(tok_seq.ids))

                # Log results
                nb_beats = round(float(mean(Tensor(nb_beats))) / mean_time, 1)
                nb_base_tokens = round(float(mean(Tensor(nb_base_tokens))) / mean_time, 1)
                nb_notes = round(float(mean(Tensor(nb_notes))) / mean_time, 1)
                logger.debug(
                    f"Mean beats/sec: {nb_beats:.1f}\n"
                    f"Mean tokens/sec: {nb_base_tokens:.1f}\n"
                    f"Mean notes/sec: {nb_notes:.1f}"
                )
                tok = baseline.tokenization
                if tok not in ["TSD", "REMI"]:
                    if tok.startswith("TSD"):
                        tok = "TSD"
                    elif tok.startswith("REMI") or tok in ["CPWord", "OctupleMono"]:
                        tok = "REMI"
                df.at[indexes[bi], f"Nb beats - {batch_size} - {tok}"] = nb_beats
                df.at[indexes[bi], f"Nb base tokens - {batch_size} - {tok}"] = nb_base_tokens
                df.at[indexes[bi], f"Nb notes - {batch_size} - {tok}"] = nb_notes

    df.to_csv(out_dir / "inference_speed_ete.csv")
    df.to_latex(out_dir / "inference_speed_ete.ltx", bold_rows=True)
