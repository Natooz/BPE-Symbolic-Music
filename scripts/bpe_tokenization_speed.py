#!/usr/bin/python3 python

"""Plots cosine sim of embeddings of BPE experiments

"""
from pathlib import Path
from time import time
import logging

import numpy as np
from miditoolkit import MidiFile
from transformers import set_seed
from tqdm import tqdm

from exp_generation import experiments
from scripts.utils import mean_std, reduce_list


if __name__ == "__main__":
    nb_files = 1000
    set_seed(777)
    (out_dir := Path("analysis")).mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("bpe_tokenization_time")
    logger_.addHandler(logging.FileHandler(out_dir / "bpe_tokenization_time.log"))
    logger_.addHandler(logging.StreamHandler())
    logger_.setLevel(logging.DEBUG)

    for exp in experiments:
        logger_.debug(f"\n{exp.name}")
        mid_paths = reduce_list(list(exp.data_path_midi.glob("**/*.mid")), nb_files)
        for baseline in exp.baselines:
            # Record tokenization and detokenization times
            times_encode, times_decode = [], []
            for file_path in tqdm(mid_paths, desc="Converting MIDIs - tokens"):
                midi = MidiFile(file_path)
                t0 = time()
                tokens = baseline.tokenizer.midi_to_tokens(midi)
                times_encode.append(time() - t0)

                t0 = time()
                _ = baseline.tokenizer.tokens_to_midi(tokens)
                times_decode.append(time() - t0)

            logger_.debug(
                f"{baseline.name} - Tokenize: {mean_std(np.array(times_encode))} | "
                f"Detokenize: {mean_std(np.array(times_decode))} "
                f"({mean_std(np.array(times_decode) * 1000)} ms)"
            )
