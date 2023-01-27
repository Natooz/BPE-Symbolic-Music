#!/usr/bin/python3 python

"""Analyzes the token types of the new BPE tokens

"""
from pathlib import Path

import pandas as pd

from exp_gen import experiments


if __name__ == '__main__':
    out_dir = Path('analysis', 'bpe_token_types')
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_typ_short = {'Pitch': 'Pch', 'Velocity': 'Vel', 'Duration': 'Dur', 'Time-Shift': 'TS', 'Position': 'Pos'}
    nb_bpe_type_successions_to_plot = 6

    # Loads tokenizers
    for exp in experiments:
        for baseline in exp.baselines:
            baseline.load_tokenizer()

    for exp in experiments:
        token_types = []
        bpe_factors = []

        # Get nb of
        for baseline in exp.baselines:
            if baseline.bpe_factor == 0:
                continue
            bpe_factors.append(baseline.bpe_factor)
            # Get distribution of token types covered by BPE tokens
            existing_types = {}
            nb_bpe_tokens = 0
            for token, event in baseline.tokenizer.vocab.token_to_event.items():
                token_type, token_val = event.split('_')
                if token_type == 'BPE':
                    sub_tok_types = token_val.split('.')[1].split('-')
                    sub_tok_types = [baseline.tokenizer.vocab.token_type(int(t)) for t in sub_tok_types]
                    sub_tok_types = '-'.join([(tok_typ_short[t] if t in tok_typ_short else t) for t in sub_tok_types])
                    try:
                        existing_types[sub_tok_types] += 1
                    except KeyError:
                        existing_types[sub_tok_types] = 1
                    nb_bpe_tokens += 1
            # Normalize and store types
            for i in existing_types.keys():
                existing_types[i] /= nb_bpe_tokens
            token_types.append(existing_types)

        # Get the most present BPE type successions
        types_prob = {}
        for i in range(len(token_types)):
            for type_, val in token_types[i].items():
                try:
                    types_prob[type_] += val
                except KeyError:
                    types_prob[type_] = val
        types_prob = dict(sorted(types_prob.items(), key=lambda kv: -kv[1])[:nb_bpe_type_successions_to_plot])

        # Keep only most present types, others are gathered as "other"
        for i in range(len(token_types)):  # sort them by token type succession / same order
            token_types_sorted = []
            for type_ in types_prob.keys():
                try:
                    token_types_sorted.append(token_types[i][type_])
                except KeyError:
                    token_types_sorted.append(0.)
            token_types_sorted.append(0.)  # For others
            for type_, val in token_types[i].items():  # add other token types prob
                if type_ not in types_prob.keys():
                    token_types_sorted[-1] += val
            token_types[i] = token_types_sorted  # replace with new numbers

        # Plots the bars using pandas
        df = pd.DataFrame([[str(bpe_factors[i])] + token_types[i] for i in range(len(token_types))],
                          columns=['BPE factor'] + list(types_prob.keys()) + ['Other'])

        plot = df.plot(x='BPE factor', kind='bar',  stacked=False, figsize=(5, 3), rot=0, width=0.8)
        plot.legend(loc='upper right', prop={'size': 8})
        plot.set_xlabel('BPE Factor')
        plot.set_ylabel('Proportion')
        fig = plot.get_figure()
        fig.savefig(out_dir / f'{exp.name}.pdf', bbox_inches='tight', dpi=300)
        fig.clf()

        '''import numpy as np
        x = np.arange(len(BPE_FACTORS))  # the label locations
        width = 0.15  # the width of the bars

        fig, ax = plt.subplots()
        for i in range(len(type_labels)):
            arr = [t[i] for t in token_types]
            rect = ax.bar(x - width / len(type_labels), arr, width, label=type_labels[i])
            ax.bar_label(rect, padding=3)
        ax.set_ylabel('%')
        # ax.set_title('')
        ax.set_xticks(x, BPE_FACTORS)
        ax.legend()

        fig.tight_layout()
        plt.savefig(out_dir / f'bpe_token_types_{group_name}.pdf', bbox_inches='tight', dpi=300)
        fig.clf()'''
