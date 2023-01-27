#!/usr/bin/python3 python

"""Analyze nb of token combinations of the learned vocabularies

"""
from pathlib import Path
from matplotlib import pyplot as plt

from torch import Tensor, cat, mean, max as max_pt
from torchtoolkit.utils import seed_everything
from tqdm import tqdm

from exp_gen import experiments


if __name__ == '__main__':
    seed_everything(777)
    (out_dir := Path('analysis')).mkdir(parents=True, exist_ok=True)
    '''logger = logging.getLogger('bpe_token_comb')
    logger.addHandler(logging.FileHandler(out_dir / 'bpe_token_comb.log'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)'''

    # Loads tokenizers
    for exp in experiments:
        for baseline in exp.baselines:
            baseline.load_tokenizer()

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    for exp in experiments:
        tokenizr = exp.baselines[5].tokenizer  # with highest bpe
        '''logger.debug(f'\n{exp.name}')'''

        tokens, bpe_comb, bpe_comb_means, bpe_comb_max = [], [], [], []
        for token, event in tqdm(tokenizr.vocab.token_to_event.items(), desc=f'Analyzing BPE tokens of {exp.name}'):
            token_type, token_val = event.split('_')
            if token_type == 'BPE':
                tokens.append(token)
                nb_combs = Tensor([len(token_val.split('.')[1].split('-'))])  # bpe-combs.prime-combs
                bpe_comb = cat([bpe_comb, nb_combs]) if isinstance(bpe_comb, Tensor) else nb_combs
                bpe_comb_means.append(mean(bpe_comb).item())
                bpe_comb_max.append(max_pt(bpe_comb).item())

        # Analysis for real data
        '''logger.debug(f'Mean nb of bpe combinations: {mean(bpe_comb):.2f} ± {std(bpe_comb):.2f}')
        logger.debug(f'Max nb of bpe combinations: {max_pt(bpe_comb):.0f}')'''
        dataset = exp.dataset if exp.dataset != 'POP909-merged' else 'POP909'
        axs[0].plot(tokens, bpe_comb_means, label=f'{dataset} {exp.baselines[0].tokenization}')
        axs[1].plot(tokens, bpe_comb_max, label=f'{dataset} {exp.baselines[0].tokenization}')

    axs[0].legend()
    for ax in axs:
        ax.grid(linestyle="-")
        ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
        ax.set_xticklabels(['0', '2k', '4k', '6k', '8k', '10k', '12k', '14k'], fontsize=9)
        ax.set_xlabel('Vocabulary size')
    axs[0].set_ylabel('Avg. token combinations')
    axs[1].set_ylabel('Max. token combinations')
    plt.savefig(out_dir / f'bpe_nb_tok_combinations.pdf', bbox_inches='tight', dpi=300)
