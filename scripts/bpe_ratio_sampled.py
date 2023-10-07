#!/usr/bin/python3 python

"""Reads BPE tokens files to measure the variations from non-BPE
Token combinations are those from the original and generated data, not the learned vocab

"""
from pathlib import Path
from typing import List, Tuple
import logging
from matplotlib import pyplot as plt
import json

import numpy as np
from miditok import MIDITokenizer
import seaborn as sns
from tqdm import tqdm

from exp_generation import experiments
from scripts.utils import get_token_dist, mean_std


def norm_sum(tens: np.ndarray) -> np.ndarray:  # by sum instead of magnitude
    return tens / np.sum(tens)


def analysis(
    tokenizer: MIDITokenizer,
    files_paths: List[Path],
    prefix_log: str,
    plot_token_dist: bool = False,
    out_path: Path = None,
) -> Tuple[np.ndarray, np.ndarray]:
    tokens_dist = get_token_dist(files_paths)
    special_token_ids = [
        tokenizer[f"{token}_None"] for token in tokenizer.special_tokens
    ]
    tokens_hist = np.histogram(tokens_dist, range(0, len(tokenizer) + 1))[0]
    idx_to_keep = [
        idx for idx in range(len(tokens_hist)) if idx not in special_token_ids
    ]
    tokens_hist = tokens_hist[idx_to_keep]
    tokens_sampled = np.where(tokens_hist != 0)[0]
    nb_sampled = len(tokens_sampled)
    voc_size = len(tokenizer) - len(special_token_ids)
    logger.debug(
        f"{prefix_log} - Nb_sampled: {nb_sampled} "
        f"({nb_sampled / voc_size * 100:.1f}\%)"
        f"| Nb_not_sampled (%): {voc_size - nb_sampled} "
        f"({(voc_size - nb_sampled) / voc_size * 100:.1f}\%)"
    )

    if tokenizer.has_bpe:
        voc_bpe_inv = {v: k for k, v in tokenizer.vocab_bpe.items()}
        bpe_combs = []
        for token in tokens_sampled:
            if token >= len(tokenizer.vocab):  # i.e. BPE token
                sub_tokens = tokenizer._vocab_bpe_bytes_to_tokens[voc_bpe_inv[token]]
                bpe_combs.append(len(sub_tokens))
        logger.debug(
            f"{prefix_log} - Mean nb bpe token combs (sampled): {mean_std(bpe_combs)})"
        )

    if plot_token_dist:
        plt.figure(figsize=(5, 3))
        sns.histplot(tokens_dist, fill=True)
        plt.savefig(out_path, bbox_inches="tight")
        plt.clf()

    return tokens_dist, tokens_hist


def analysis_embed_pooling(tokenizer: MIDITokenizer, files_paths: List[Path], prefix_log: str):
    # one dist per sub vocab
    tokens_dists = [[] for _ in range(len(tokenizer.vocab))]
    for file_path in tqdm(files_paths, "loading files"):
        with open(file_path) as json_file:
            tokens = json.load(json_file)["ids"]
        for i, seq in enumerate(tokens):
            tokens_dists[i] += seq

    all_nb_sampled = 0
    for i in range(len(tokens_dists)):
        tokens_hist = np.histogram(np.array(tokens_dists[i]), range(0, len(tokenizer.vocab[i]) + 1))[0]
        idx_to_keep = list(range(5, len(tokenizer.vocab[i])))
        tokens_hist = tokens_hist[idx_to_keep]

        tokens_sampled = np.where(tokens_hist != 0)[0]
        all_nb_sampled += len(tokens_sampled)

    voc_size = len(tokenizer) - len(tokenizer.vocab) * 5  # because 5 special tokens
    logger.debug(
        f"{prefix_log} - Nb_sampled: {all_nb_sampled} "
        f"({all_nb_sampled / voc_size * 100:.1f}\%)"
        f"| Nb_not_sampled (%): {voc_size - all_nb_sampled} "
        f"({(voc_size - all_nb_sampled) / voc_size * 100:.1f}\%)"
    )


if __name__ == "__main__":
    (out_dir := Path("analysis", "ratio_sampled_gen")).mkdir(
        parents=True, exist_ok=True
    )
    logger = logging.getLogger("ratio_sampled")
    logger.addHandler(logging.FileHandler(out_dir / "ratio_sampled_mean_nb_comb.log"))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    for exp in experiments:
        for baseline in exp.baselines:
            logger.debug(
                f"\n{exp.name} - {baseline.name} - vocab size: {len(baseline.tokenizer)} tokens"
            )
            if baseline.tokenizer.is_multi_voc:
                analysis_embed_pooling(baseline.tokenizer, list((baseline.run_path / "gen").glob("**/*.json")), "GEN")
                continue

            data_files_paths = list(baseline.tokens_path.glob("**/*.json"))
            gen_files_paths = list((baseline.run_path / "gen").glob("**/*.json"))

            # Analysis for real data
            _, tokens_hist_data = analysis(
                baseline.tokenizer,
                data_files_paths,
                "DATA",
                False,
                out_dir / f"token_dist_{exp.name}_{baseline.name}_DATA.pdf",
            )

            # Analysis for generated data
            _, tokens_hist_gen = analysis(
                baseline.tokenizer,
                gen_files_paths,
                "GEN",
                False,
                out_dir / f"token_dist_{exp.name}_{baseline.name}_GEN.pdf",
            )
            tokens_hist_data, tokens_hist_gen = norm_sum(tokens_hist_data), norm_sum(
                tokens_hist_gen
            )
            oa = sum(
                [
                    min(tokens_hist_data[i], tokens_hist_gen[i])
                    for i in range(len(tokens_hist_gen))
                ]
            )
            logger.debug(f"OA dist tokens DATA-GEN: {oa:.4f}")
