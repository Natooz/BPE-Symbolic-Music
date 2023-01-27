#!/usr/bin/python3 python

"""Reads BPE tokens files to measure the variations from non-BPE
Token combinations are those from the original and generated data, not the learned vocab

"""
from pathlib import Path
from typing import List, Tuple
import logging
from matplotlib import pyplot as plt

import numpy as np
from miditok import MIDITokenizer
import seaborn as sns

from exp_gen import experiments
from scripts.utils import get_token_dist, mean_std


def norm_sum(tens: np.ndarray) -> np.ndarray:  # by sum instead of magnitude
    return tens / np.sum(tens)


def analysis(tokenizer: MIDITokenizer, files_paths: List[Path], prefix_log: str, bpe: bool,
             plot_token_dist: bool = False, out_path: Path = None) -> Tuple[np.ndarray, np.ndarray]:
    tokens_dist = get_token_dist(files_paths)
    tokens_hist = np.histogram(tokens_dist, range(0, len(tokenizer) + 1))[0]
    tokens_sampled = np.where(tokens_hist != 0)[0]
    nb_sampled = len(tokens_sampled)
    logger.debug(f'{prefix_log} - Nb_sampled | nb_not_sampled (%): '
                 f'{nb_sampled} $\lvert$ {len(tokenizer) - nb_sampled} '
                 f'({int((len(tokenizer) - nb_sampled) / len(baseline.tokenizer) * 100)}\%)')

    if bpe:
        bpe_combs = []
        for token in tokens_sampled:
            token_type, token_val = tokenizer[int(token)].split('_')
            if token_type == 'BPE':
                bpe_combs.append(len(token_val.split('.')[1].split('-')))  # bpe-combs.prime-combs
        logger.debug(f'{prefix_log} - Nb bpe token combs (sampled): {mean_std(bpe_combs)})')

    if plot_token_dist:
        plt.figure(figsize=(5, 3))
        sns.histplot(tokens_dist, fill=True)
        plt.savefig(out_path, bbox_inches='tight')
        plt.clf()

    return tokens_dist, tokens_hist


if __name__ == '__main__':
    (out_dir := Path('analysis', 'ratio_sampled_gen')).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('ratio_sampled')
    logger.addHandler(logging.FileHandler(out_dir / 'ratio_sampled_mean_nb_comb.log'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    # Loads tokenizers
    for exp in experiments:
        for baseline in exp.baselines:
            baseline.load_tokenizer()

    for exp in experiments:
        for baseline in exp.baselines:
            if baseline.is_embed_pooling:
                continue  # embed pooling not concerned
            logger.debug(f'\n{exp.name} - {baseline.name} - vocab size: {len(baseline.tokenizer)} tokens')
            data_files_paths = list(baseline.data_path.glob('**/*.json'))
            generated_files_paths = list(baseline.gen_data_path.glob('**/*.json'))

            # Analysis for real data
            _, tokens_hist_data = analysis(baseline.tokenizer, data_files_paths, 'DATA',
                                           True if baseline.bpe_factor > 0 else False, False,
                                           out_dir / f'token_dist_{exp.name}_{baseline.name}_DATA.pdf')

            # Analysis for generated data
            _, tokens_hist_gen = analysis(baseline.tokenizer, generated_files_paths, 'GEN',
                                          True if baseline.bpe_factor > 0 else False, False,
                                          out_dir / f'token_dist_{exp.name}_{baseline.name}_GEN.pdf')
            tokens_hist_data, tokens_hist_gen = norm_sum(tokens_hist_data), norm_sum(tokens_hist_gen)
            oa = sum([min(tokens_hist_data[i], tokens_hist_gen[i]) for i in range(len(tokens_hist_gen))])
            logger.debug(f'OA dist tokens DATA-GEN: {oa:.4f}')
