#!/usr/bin/python3 python

"""Analyzes the token types of the new BPE tokens

"""
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from exp_generation import experiments as exp_gen
from exp_cla_finetune import experiments as exp_cla
from constants import BPE_VOCAB_SIZES


def plot_token_types(df: pd.DataFrame, file_name: str):
    plot = df.plot(
        x="Vocab size", kind="bar", stacked=False, figsize=(5, 3), rot=0, width=0.8
    )
    plot.legend(loc="upper right", prop={"size": 8})
    plot.set_xlabel("Vocab size")
    plot.set_ylabel("Proportion")
    fig = plot.get_figure()
    extent = (
        plt.gca()
        .get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .translated(-0.25, -0.2)
        .expanded(1.16, 1.2)
    )
    fig.savefig(out_dir / file_name, bbox_inches=extent, dpi=300)
    fig.clf()


if __name__ == "__main__":
    out_dir = Path("analysis", "bpe_token_types")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_typ_short = {
        "Pitch": "Pit",
        "Velocity": "Vel",
        "Duration": "Dur",
        "TimeShift": "TS",
        "Position": "Pos",
        "Program": "Prg",
    }
    nb_bpe_type_successions_to_plot = 6

    data_tokenization_already_treated = []
    all_token_types = {s: {} for s in BPE_VOCAB_SIZES}

    for exp in exp_gen + exp_cla:
        dataset = exp.dataset if exp.dataset != "POP909-merged" else "POP909"
        data_tok = f"{dataset} {exp.baselines[0].tokenization}"
        if data_tok in data_tokenization_already_treated:
            continue

        token_types = []
        bpe_vocab_sizes = []

        # Get nb of
        for baseline in exp.baselines:
            if baseline.tokenization_config.bpe_vocab_size is None:
                continue
            bpe_vocab_sizes.append(
                str(int(baseline.tokenization_config.bpe_vocab_size / 1000)) + "k"
            )
            # Get distribution of token types covered by BPE tokens
            existing_types = {}
            nb_bpe_tokens = 0
            for bytes_, tokens in baseline.tokenizer._vocab_bpe_bytes_to_tokens.items():
                if len(tokens) > 1:
                    sub_tok_types = [tok.split("_")[0] for tok in tokens]
                    sub_tok_types = "-".join(
                        [
                            (tok_typ_short[t] if t in tok_typ_short else t)
                            for t in sub_tok_types
                        ]
                    )
                    try:
                        existing_types[sub_tok_types] += 1
                        all_token_types[baseline.tokenization_config.bpe_vocab_size][
                            sub_tok_types
                        ] += 1
                    except KeyError:
                        existing_types[sub_tok_types] = 1
                        all_token_types[baseline.tokenization_config.bpe_vocab_size][
                            sub_tok_types
                        ] = 1
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
        types_prob = dict(
            sorted(types_prob.items(), key=lambda kv: -kv[1])[
                :nb_bpe_type_successions_to_plot
            ]
        )

        # Keep only most present types, others are gathered as "other"
        for i in range(
            len(token_types)
        ):  # sort them by token type succession / same order
            token_types_sorted = []
            for type_ in types_prob.keys():
                try:
                    token_types_sorted.append(token_types[i][type_])
                except KeyError:
                    token_types_sorted.append(0.0)
            token_types_sorted.append(0.0)  # For others
            for type_, val in token_types[i].items():  # add other token types prob
                if type_ not in types_prob.keys():
                    token_types_sorted[-1] += val
            token_types[i] = token_types_sorted  # replace with new numbers

        # Plots the bars using pandas
        df_ = pd.DataFrame(
            [
                [str(bpe_vocab_sizes[i])] + token_types[i]
                for i in range(len(token_types))
            ],
            columns=["Vocab size"] + list(types_prob.keys()) + ["Other"],
        )
        plot_token_types(df_, f"{'_'.join(exp.name.split('_')[-2:])}.pdf")
        data_tokenization_already_treated.append(data_tok)

    # Do it for all token types combined
    """combined_types = {}
    for voc_size, all_toks in all_token_types.items():
        for type_, nb_occurrences in all_toks.items():
            parts = type_.split("-")
            if len(parts) > 3 and ("Pos" in parts or "TS" in parts):
                for i, part in enumerate(parts):
                    if part in ["Pos", "TS"]:
                        parts[i] = "Pos/TS"
                        break
                combined = "-".join(parts)
                try:
                    combined_types[combined] += [type_]
                except KeyError:
                    combined_types[combined] = [type_]

    for combined_type, types_to_replace in combined_types.items():
        for voc_size, all_toks in all_token_types.items():
            if len(types_to_replace) > 1:
                all_token_types[voc_size][combined_type] = 0
                for typ in types_to_replace:
                    if typ in all_token_types[voc_size]:
                        all_token_types[voc_size][combined_type] += all_token_types[voc_size].pop(typ)

    # Get the most present BPE type successions
    types_prob = {}
    for voc_size in all_token_types:
        for type_, val in all_token_types[voc_size].items():
            try:
                types_prob[type_] += val
            except KeyError:
                types_prob[type_] = val
    types_prob = dict(sorted(types_prob.items(), key=lambda kv: -kv[1])[:nb_bpe_type_successions_to_plot])

    # Keep only most present types, others are gathered as "other"
    for voc_size in all_token_types:  # sort them by token type succession / same order
        token_types_sorted = []
        for type_ in types_prob.keys():
            try:
                token_types_sorted.append(all_token_types[voc_size][type_])
            except KeyError:
                token_types_sorted.append(0.)
        token_types_sorted.append(0.)  # For others
        for type_, val in all_token_types[voc_size].items():  # add other token types prob
            if type_ not in types_prob.keys():
                token_types_sorted[-1] += val
        all_token_types[voc_size] = token_types_sorted  # replace with new numbers

    # Plots the bars using pandas
    df_ = pd.DataFrame([[str(voc_size)] + all_token_types[voc_size] for voc_size in all_token_types],
                       columns=['Vocab size'] + list(types_prob.keys()) + ['Other'])
    plot_token_types(df_, f"all_combined.pdf")"""
