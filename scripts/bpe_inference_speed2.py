#!/usr/bin/python3 python

from pathlib import Path
from time import time
import logging

from torch import Tensor, LongTensor, no_grad, mean, sum as sum_pt, isin
from torch.utils.data import DataLoader
from transformers.trainer_utils import set_seed, get_last_checkpoint
from tqdm import tqdm
from miditok import TokSequence
from pandas import DataFrame

from dataset import DatasetMIDI, DataCollatorStandard
from exp_generation import experiments
from training import select_device


if __name__ == "__main__":
    (out_dir := Path("analysis")).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("inference_speed2")
    logger.addHandler(logging.FileHandler(out_dir / "inference_speed2.log"))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    set_seed(444)
    NB_FILES = 10
    indexes = ["No BPE", "BPE 1k", "BPE 5k", "BPE 10k", "BPE 20k", "PVm", "PVDm", "CPWord", "Octuple"]
    metrics = ["Nb beats", "Nb base tokens", "Nb notes"]
    df = DataFrame(columns=[f"{m} - {tok}" for m in metrics for tok in ["TSD", "REMI"]], index=indexes)

    for exp in experiments:
        for bi, baseline in enumerate(exp.baselines):
            logger.debug(
                f"\n{exp.name} - {baseline.name} - vocab size: {len(baseline.tokenizer)} tokens"
            )
            data_files_paths = list(baseline.tokens_path.glob("**/*.json"))[:NB_FILES]

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

            # Loading data
            dataset = DatasetMIDI(
                data_files_paths,
                baseline.data_config.min_seq_len - 1,
                baseline.data_config.max_seq_len - 1,
                baseline.tokenizer,
            )
            collator = DataCollatorStandard(
                baseline.pad_token, baseline.bos_token, pad_on_left=True
            )
            collator.pad_on_left = True
            dataloader = DataLoader(
                dataset, baseline.test_config.batch_size, collate_fn=collator
            )

            # Generate
            times = []
            all_gen_tokens = []
            with no_grad():
                for batch_ in tqdm(
                    dataloader, desc=f"Measuring inference speed {baseline.name}"
                ):
                    batch_size, prompt_len = batch_["input_ids"].shape[:2]
                    for key in batch_:
                        if isinstance(batch_[key], Tensor):
                            batch_[key] = batch_[key].to(device)
                    t0 = time()
                    generated_tokens = model_.generate(
                        **batch_, generation_config=baseline.generation_config
                    )  # (N,T)
                    t1 = time() - t0
                    times += [t1] * batch_size
                    for seq in generated_tokens.cpu().tolist():
                        all_gen_tokens.append(seq[prompt_len:])

            # Decode tokens and compute inference speeds
            nb_beats, nb_base_tokens, nb_notes = [], [], []
            for seq, time_gen in zip(all_gen_tokens, times):
                midi = baseline.tokenizer(
                    [seq] if not baseline.tokenizer.unique_track else seq
                )
                nb_beats.append(midi.max_tick / midi.ticks_per_beat)
                nb_notes.append(len(midi.instruments[0].notes))
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
            mean_time = sum(times) / len(times)
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
            df.at[indexes[bi], f"Nb beats - {tok}"] = nb_beats
            df.at[indexes[bi], f"Nb base tokens - {tok}"] = nb_base_tokens
            df.at[indexes[bi], f"Nb notes - {tok}"] = nb_notes

    df.to_csv(out_dir / "inference_speed2.csv")
    df.to_latex(out_dir / "inference_speed2.ltx", bold_rows=True)
