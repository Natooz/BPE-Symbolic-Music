#!/usr/bin/python3 python

"""Analyze nb of token combinations of the learned vocabularies

"""
from pathlib import Path
from time import time

from miditoolkit import MidiFile
from torch import Tensor, cat, mean, max as max_pt
from transformers import set_seed
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pandas import DataFrame

from exp_generation import experiments as exp_gen
from exp_cla_finetune import experiments as exp_cla
from dataset import list_mmd_files_paths
from scripts.tokenize_datasets import is_midi_valid, is_midi_valid_multi
from scripts.utils import reduce_list


if __name__ == "__main__":
    limit_nb_midi_files = 500
    id_markers = ["o", "v", "^", "x", "p", "*"]
    set_seed(777)
    (out_dir := Path("analysis", "nb_tok")).mkdir(parents=True, exist_ok=True)

    # Create figure and set axis labels
    fig, axs = plt.subplots(1, 5, figsize=(25, 3))
    axs[0].set_ylabel("Avg. token comb.", fontsize=14)
    axs[1].set_ylabel("Max. token comb.", fontsize=14)
    for ax in axs[:2]:
        ax.grid(linestyle="-")
        ax.set_xticks([0, 5000, 10000, 15000, 20000])
        ax.set_xticklabels(["0", "5k", "10k", "15k", "20k"], fontsize=13)
        ax.set_xlabel("Vocabulary size", fontsize=14)

    xticks_labels = [
        baseline.name if baseline.name != "OctupleMono" else "Octuple"
        for baseline in exp_gen[1].baselines
        if not baseline.name.endswith("tiny")
    ]
    for i in range(len(xticks_labels)):
        new_label = xticks_labels[i].split("_")[-1]
        if new_label.startswith("bpe"):
            new_label = new_label[:-3] + "k"
        elif new_label.startswith(exp_gen[1].baselines[0].tokenization):
            new_label = new_label[len(exp_gen[1].baselines[0].tokenization) :]
        xticks_labels[i] = new_label
    xticks = range(len(xticks_labels))
    metric_names = ["avg_nb_comb", "max_nb_comb", "avg_tpb", "tok_time", "detok_time"]

    axs[2].set_ylabel("Tokens per beat (tpb)", fontsize=14)
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticks_labels, rotation=72)
    axs[2].grid("on")

    axs[3].set_ylabel("Time (sec)", fontsize=14)
    axs[3].set_xticks(xticks)
    axs[3].set_xticklabels(xticks_labels, rotation=72)
    axs[3].grid("on")
    axs[3].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    axs[4].set_ylabel("Time (sec)", fontsize=14)
    axs[4].set_xticks(xticks)
    axs[4].set_xticklabels(xticks_labels, rotation=72)
    axs[4].grid("on")
    axs[4].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    data_tokenization_already_treated = []
    marker_id = 0
    row_names = ["No BPE", "BPE 1k", "BPE 5k", "BPE 10k", "BPE 20k", "PVm", "PVDm", "CP Word", "Octuple"]
    column_names = [f"{metric} {tok}" for metric in metric_names[2:] for tok in ["TSD", "REMI"]]
    dfs = {dataset: DataFrame(index=row_names, columns=column_names) for dataset in ["Maestro", "MMD"]}
    for exp in exp_gen + exp_cla:
        data_tok = f"{exp.dataset} {exp.baselines[0].tokenization}"
        if data_tok in data_tokenization_already_treated:
            continue

        tokenizr = exp.baselines[4].tokenizer  # with highest bpe vocab

        # Average and maximum nb of token combinations
        ids, bpe_comb, bpe_comb_means, bpe_comb_max = [], [], [], []
        sorted_bpe_voc = dict(
            sorted(tokenizr.vocab_bpe.items(), key=lambda item: item[1])
        )
        for bytes_, token_id in tqdm(
            sorted_bpe_voc.items(), desc=f"Analyzing BPE tokens of {exp.name}"
        ):
            tokens = tokenizr._vocab_bpe_bytes_to_tokens[bytes_]
            if len(tokens) > 1:
                ids.append(token_id)  # id, vocab might me not sorted by id
                nb_combs = Tensor([len(tokens)])
                bpe_comb = (
                    cat([bpe_comb, nb_combs])
                    if isinstance(bpe_comb, Tensor)
                    else nb_combs
                )
                bpe_comb_means.append(mean(bpe_comb).item())
                bpe_comb_max.append(max_pt(bpe_comb).item())

        # Tokens per beat (tpb)
        if exp.dataset != "MMD":
            midi_paths = list(exp.data_path_midi.glob("**/*.mid"))
            valid_fn = is_midi_valid
        else:
            midi_paths = list_mmd_files_paths(
                Path("data", "MMD_METADATA", "midi_audio_matches.json")
            )
            valid_fn = is_midi_valid_multi
        midi_paths = reduce_list(midi_paths, limit_nb_midi_files)
        avg_tpb, avg_tok, avg_detok = [], [], []
        for bi, baseline in enumerate(tqdm(
            exp.baselines, desc=f"Computing tok - detok times ({exp.name})"
        )):
            if baseline.name.endswith("tiny"):
                continue
            tpb, tok_times, detok_times = [], [], []
            for midi_path in midi_paths:
                try:
                    midi = MidiFile(midi_path)
                except:
                    continue
                if not valid_fn(midi):
                    continue
                t0 = time()
                tokens = baseline.tokenizer.midi_to_tokens(midi)
                tok_time = time() - t0

                if len(tokens) == 0:
                    continue

                tok_times.append(tok_time)
                nb_tokens = len(tokens.ids if tokenizr.unique_track else tokens[0].ids)
                tpb.append(nb_tokens / (midi.max_tick / midi.ticks_per_beat))

                t0 = time()
                _ = baseline.tokenizer.tokens_to_midi(tokens)
                detok_times.append(time() - t0)

            avg_tpb.append(sum(tpb) / len(tpb))
            avg_tok.append(sum(tok_times) / len(tok_times))
            avg_detok.append(sum(detok_times) / len(detok_times))
            offset = 1 if exp.name.endswith("REMI") or exp.name.endswith("REMIPlus") else 0
            row_offset = 1 if baseline.tokenization == "Octuple" else 0
            tpb = round(avg_tpb[-1], 1)
            if bi > 0:
                tpb_variation = round((avg_tpb[-1] - avg_tpb[0]) / avg_tpb[0] * 100, 2)
                tpb = f"{tpb} ({tpb_variation:.1f}\%)"
            dfs[exp.dataset].at[row_names[bi + row_offset], column_names[offset]] = tpb
            dfs[exp.dataset].at[row_names[bi + row_offset], column_names[2 + offset]] = round(avg_tok[-1], 3)
            dfs[exp.dataset].at[row_names[bi + row_offset], column_names[4 + offset]] = round(avg_detok[-1], 3)

        # Plot on axes
        axs[0].plot(ids, bpe_comb_means, label=data_tok)
        axs[1].plot(ids, bpe_comb_max, label=data_tok)
        if exp.dataset == "MMD" and exp.name.endswith("REMIPlus"):
            ticks = list(range(len(avg_tpb) - 1)) + [len(avg_tpb)]
        else:
            ticks = list(range(len(avg_tpb)))
        axs[2].scatter(ticks, avg_tpb, label=data_tok, marker=id_markers[marker_id])
        axs[3].scatter(ticks, avg_tok, label=data_tok, marker=id_markers[marker_id])
        axs[4].scatter(ticks, avg_detok, label=data_tok, marker=id_markers[marker_id])

        data_tokenization_already_treated.append(data_tok)
        marker_id += 1

    # Save raw metrics
    for dataset, df in dfs.items():
        df.to_csv(out_dir / f"tok_comb_speed_{dataset}.csv")
        df.to_latex(out_dir / f"tok_comb_speed_{dataset}.txt", bold_rows=True)

    axs[0].legend()

    for i, name in enumerate(metric_names):
        # extent = full_extent(axs[i]).transformed(fig.dpi_scale_trans.inverted())
        extent = (
            axs[i]
            .get_window_extent()
            .transformed(fig.dpi_scale_trans.inverted())
            .translated(-0.275, -0.3)
            .expanded(1.2, 1.3)
        )
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches=extent, dpi=300)
