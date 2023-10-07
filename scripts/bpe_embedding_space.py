#!/usr/bin/python3 python

"""Plots cosine sim of embeddings of BPE experiments

"""
from pathlib import Path
from typing import List, Dict

import miditok
import numpy as np
import scipy.linalg
from torch import (
    Tensor,
    LongTensor,
    cat,
    no_grad,
    mean,
    var,
    multinomial,
    arange,
    from_numpy,
    zeros,
    full,
    triu,
    cosine_similarity,
    std,
    abs as abs_pt,
)
from transformers import set_seed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.interpolate import UnivariateSpline
from umap import UMAP
from skdim.id import lPCA, MOM, FisherS
import faiss
from pandas import DataFrame, to_numeric

assert hasattr(faiss, "IndexFlatIP")

from classes import Baseline
from metrics import isoscore
from training import select_device


nb_bpe_type_successions_to_plot = 5
tok_typ_short = {
    "Pitch": "Pi",
    "Velocity": "Ve",
    "Duration": "Du",
    "TimeShift": "TS",
    "Position": "Po",
    "Program": "Pr",
}
FAISS_SEARCH_BATCH_SIZE = 100
FAISS_TOP_K = 64
MAX_NB_EMBEDS = 50000
MAX_NB_POINTS_UMAP = 4000
id_func = [lPCA, MOM, FisherS]
id_markers = ["o", "x", "*"]
SECOND_AXIS_LIM = 80  # for lPCA


def compute_embedding_distances(embeddings: Tensor, out_path: Path):
    # Euclidian distance and cosine similarity
    # (https://aclanthology.org/D19-1006/)
    # (A Contrastive Framework for Neural Text Generation)
    # We do not batch compute them as it requires too much memory
    # pw_eucl_dis = cdist(embeddings, embeddings)  # batched
    # pw_cos_sim = cosine_similarity(embeddings[:, :, None], embeddings.t()[None, :, :]).cpu()
    pw_eucl_dis = zeros((embeddings.shape[0], embeddings.shape[0])).to(embeddings.device)
    pw_cos_sim = zeros((embeddings.shape[0], embeddings.shape[0])).to(embeddings.device)
    for i in range(embeddings.shape[0]):  # non batched
        for j in range(i, embeddings.shape[0]):  # compute full matrix as we will plot cosine sim
            eucl_dist = (embeddings[i] - embeddings[j]).pow(2).sum().sqrt().unsqueeze(0)
            cos_sim = cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
            pw_eucl_dis[i, j] = eucl_dist
            pw_eucl_dis[j, i] = eucl_dist
            pw_cos_sim[i, j] = cos_sim
            pw_cos_sim[j, i] = cos_sim
    triu_mask = triu(full(pw_cos_sim.shape, 1).bool(),  diagonal=1)
    pw_eucl_dis_vals = pw_eucl_dis[triu_mask]
    pw_cos_sim_vals = abs_pt(pw_cos_sim[triu_mask])
    mean_pw_dis, std_pw_dis, var_pw_dis = mean(pw_eucl_dis_vals), std(pw_eucl_dis_vals), var(pw_eucl_dis_vals)
    mean_pw_cs, std_pw_cs, var_pw_cs = mean(pw_cos_sim_vals), std(pw_cos_sim_vals), var(pw_cos_sim_vals)
    print(f'Mean euclidian distance: {mean_pw_dis:.2f} +- {std_pw_dis:.2f} - var: {var_pw_dis:.2f}\n'
          f'Mean abs pairwise cos sim: {mean_pw_cs:.2f} +- {std_pw_cs:.2f} - var: {var_pw_cs:.2f}')

    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.matshow(pw_cos_sim.cpu(), interpolation='nearest')
    divider = make_axes_locatable(ax)  # to have the colorbar of the same height of the plot
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(im, cax=cax)
    ax.grid(True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    fig.clf()


def highlight_df_latex(dataframe: DataFrame, min_: bool = False) -> DataFrame:
    df_copy = to_numeric(dataframe.copy(deep=True))
    for column in df_copy.columns:
        if min_:
            row_idx_to_highlight = df_copy[column].argmin()
        else:
            row_idx_to_highlight = df_copy[column].argmax()
        df_copy.at[row_idx_to_highlight, column] = f"\\textbf{{{df_copy.at[row_idx_to_highlight, column]}}}"

    return df_copy


def compute_mean_var_embeds(
    embeddings: Tensor, fast_num_cells_in_search: int = 10
) -> float:
    device = embeddings.device
    embeddings = embeddings.cpu().numpy().astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])

    # using GPU version of faiss
    if device.type == "cuda" and hasattr(faiss, "StandardGpuResources"):
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(embeddings)
    index.nprobe = min(fast_num_cells_in_search, embeddings.shape[0])

    all_distances = []
    for i in range(0, len(embeddings), FAISS_SEARCH_BATCH_SIZE):
        embeds = embeddings[i : i + FAISS_SEARCH_BATCH_SIZE]
        # embeds /= embeds.norm(dim=1, keepdim=True)  # normalize_to_unit
        distances, idx = index.search(embeds, FAISS_TOP_K)  # (N,K) for both
        all_distances.append(from_numpy(distances))

    mean_var = mean(var(cat(all_distances), dim=1))
    return float(mean_var)


def compute_intrinsic_dimensions(embeddings: Tensor) -> List[int]:
    embed_np = embeddings.cpu().numpy()
    scores = []

    for func in id_func:
        id_ = func()
        res, id_score = "error", None
        try:
            id_ = id_.fit(embed_np)
            res, id_score = f"{id_.dimension_:.2f}", id_.dimension_
        except ValueError:
            res = "Value error (nan or inf)"
        finally:
            print(f"Intrinsic dimension - {id_.__class__.__name__}: {res}")
            scores.append(id_score)

    return scores


def plot_points(points: np.ndarray, token_types: dict, out_path: Path):
    # Plots by token type
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    for token_type, indices in token_types.items():
        plt.scatter(*points[indices].T, label=token_type, s=4)
        """if baseline.bpe_factor == 0:  # display value of each points
            for token in indices:
                txt = baseline.tokenizer[token].split('_')[1]
                ax.annotate(txt, t_sne.embedding_[token])"""
    plt.legend().set_zorder(100)
    # plt.savefig(out_path, bbox_inches="tight", dpi=300)
    extent = (
        ax
        .get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .translated(-0.15, -0.15)
        .expanded(1.14, 1.1)
    )
    fig.savefig(out_path, bbox_inches=extent, dpi=300)
    fig.clf()


def plot_points_3d(points, token_types: dict, out_path: Path):
    # Plots by token type
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for token_type, indices in token_types.items():
        ax.scatter(*points[indices].T, label=token_type)
    # Then BPE by type successions + others
    plt.legend(prop={"size": 8}).set_zorder(100)
    # plt.savefig(out_path, bbox_inches="tight", dpi=300)
    extent = (
        ax
        .get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        .translated(0.15, -0.05)
        .expanded(0.82, 1.08)
    )
    fig.savefig(out_path, bbox_inches=extent, dpi=300)
    fig.clf()


def get_token_types(baseline: Baseline, token_ids: List[int]) -> Dict[str, List[int]]:
    # Handles labels for the legend of UMAP plot
    # Compute recurrent BPE token type successions, and only display the most
    # recurrent ones, others are showed as "Other"

    token_types = {"Special": []}
    for plot_id, tok_id in enumerate(token_ids):
        if baseline.tokenizer.has_bpe and tok_id >= len(baseline.tokenizer.vocab):
            tok_typ = "BPE"
        else:
            tok_typ = baseline.tokenizer[tok_id].split("_")[0]
        if tok_typ in baseline.tokenizer.special_tokens:
            tok_typ = "Special"
        try:
            token_types[tok_typ].append((tok_id, plot_id))
        except KeyError:
            token_types[tok_typ] = [(tok_id, plot_id)]

    if baseline.tokenizer.has_bpe:
        bpe_types_successions = {}
        bytes_to_toks = (
            baseline.tokenizer._vocab_bpe_bytes_to_tokens
        )  # byte(s) -> token(s)
        voc_bpe_inv = {v: k for k, v in baseline.tokenizer.vocab_bpe.items()}
        for bpe_tok_id, plot_id in token_types["BPE"]:
            byte_ = voc_bpe_inv[bpe_tok_id]
            toks = bytes_to_toks[byte_]
            sub_tok_types = [tok.split("_")[0] for tok in toks]
            sub_tok_types = "-".join(
                [(tok_typ_short[t] if t in tok_typ_short else t) for t in sub_tok_types]
            )
            try:
                bpe_types_successions[sub_tok_types] += [(bpe_tok_id, plot_id)]
            except KeyError:
                bpe_types_successions[sub_tok_types] = [(bpe_tok_id, plot_id)]
        bpe_types_successions = dict(
            sorted(bpe_types_successions.items(), key=lambda i: -len(i[1]))
        )
        bpe_types_successions["Other BPE"] = []
        for key, ids in list(bpe_types_successions.items())[
            nb_bpe_type_successions_to_plot:
        ]:
            if key == "Other BPE":
                break
            bpe_types_successions["Other BPE"] += ids
            del bpe_types_successions[key]
        token_types.update(
            bpe_types_successions
        )  # update dict with decomposed BPE types
        del token_types["BPE"]  # remove composed BPE tokens, don't need it anymore

    # only keeps indexes for plotting
    for tok_typ in token_types:
        token_types[tok_typ] = [i[1] for i in token_types[tok_typ]]

    return token_types


class TokenIterator:
    def __init__(
        self, batch_size: int, tokenizer: miditok.MIDITokenizer, max_nb_embed: int
    ):
        """Training iterator class.
        Can work in two modes:
            1. Number of steps: will be iterated a fixed number of times
            2. Min valid accuracy: will be iterated till the model reaches a target validation
                accuracy value, or if the number of training steps exceeds max_nb_steps.

        :param batch_size: batch size
        :param tokenizer: tokenizer
        :param max_nb_embed: maximum nb of token embeddings to analyze (selected by ascending order in the voc)
        """
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        if tokenizer.is_multi_voc:
            if isinstance(tokenizer, (miditok.OctupleMono, miditok.Octuple)):
                self.pitches = [
                    idx
                    for tok, idx in tokenizer.vocab[0].items()
                    if tok.split("_")[0] == "Pitch"
                ]
                self.velocities = [
                    idx
                    for tok, idx in tokenizer.vocab[1].items()
                    if tok.split("_")[0] == "Velocity"
                ]
                self.duration = [
                    idx
                    for tok, idx in tokenizer.vocab[2].items()
                    if tok.split("_")[0] == "Duration"
                ]
            elif isinstance(
                tokenizer, miditok.CPWord
            ):  # only consider Pitch / Vel / Dur combinations
                self.pitches = [
                    idx
                    for tok, idx in tokenizer.vocab[2].items()
                    if tok.split("_")[0] == "Pitch"
                ]
                self.velocities = [
                    idx
                    for tok, idx in tokenizer.vocab[3].items()
                    if tok.split("_")[0] == "Velocity"
                ]
                self.duration = [
                    idx
                    for tok, idx in tokenizer.vocab[4].items()
                    if tok.split("_")[0] == "Duration"
                ]

            self.nb_samples = (
                len(self.pitches) * len(self.velocities) * len(self.duration)
            )
            if self.nb_samples > max_nb_embed:
                self.samples_idx = (
                    multinomial(arange(self.nb_samples).float(), max_nb_embed)
                    .long()
                    .sort()
                )
                self.nb_samples = len(self.samples_idx)
            else:
                self.samples_idx = arange(self.nb_samples)
        else:
            self.nb_samples = len(tokenizer)
        self.nb_samples = min(self.nb_samples, max_nb_embed)

    def __getitem__(self, idx: int) -> LongTensor:
        if not isinstance(self.tokenizer.vocab, list):
            return LongTensor([idx])
        current_pitch_idx = idx // (len(self.duration) * len(self.velocities))
        current_vel_idx = (idx // len(self.duration)) % len(self.velocities)
        current_dur_idx = idx % len(self.duration)
        if isinstance(self.tokenizer, miditok.OctupleMono):
            token = [
                self.pitches[current_pitch_idx],
                self.velocities[current_vel_idx],
                self.duration[current_dur_idx],
                self.tokenizer.vocab[3]["Position_0"],
                self.tokenizer.vocab[4]["Bar_0"],
            ]
        elif isinstance(self.tokenizer, miditok.Octuple):
            token = [
                self.pitches[current_pitch_idx],
                self.velocities[current_vel_idx],
                self.duration[current_dur_idx],
                self.tokenizer.vocab[4]["Program_0"],
                self.tokenizer.vocab[4]["Position_0"],
                self.tokenizer.vocab[5]["Bar_0"],
            ]
        else:  # CPWord
            token = [
                self.tokenizer.vocab[0]["Family_Note"],
                self.tokenizer.vocab[1]["Position_Ignore"],
                self.pitches[current_pitch_idx],
                self.velocities[current_vel_idx],
                self.duration[current_dur_idx],
            ]

        return LongTensor(token)

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self) -> LongTensor:
        if self.step >= self.nb_samples:
            raise StopIteration

        if isinstance(self.tokenizer.vocab, list):
            sample = []  # (T,Z)
            nb_vel_dur_comb = len(self.duration) * len(self.velocities)
            for s in range(
                self.step, min([self.step + self.batch_size, self.nb_samples])
            ):
                comb_idx = self.samples_idx[s]
                current_pitch_idx = comb_idx // nb_vel_dur_comb
                current_vel_idx = (comb_idx // len(self.duration)) % len(
                    self.velocities
                )
                current_dur_idx = comb_idx % len(self.duration)
                if isinstance(self.tokenizer, miditok.OctupleMono):
                    sample.append(
                        [
                            self.pitches[current_pitch_idx],
                            self.velocities[current_vel_idx],
                            self.duration[current_dur_idx],
                            self.tokenizer.vocab[3]["Position_0"],
                            self.tokenizer.vocab[4]["Bar_0"],
                        ]
                    )
                if isinstance(self.tokenizer, miditok.Octuple):
                    sample.append(
                        [
                            self.pitches[current_pitch_idx],
                            self.velocities[current_vel_idx],
                            self.duration[current_dur_idx],
                            self.tokenizer.vocab[3]["Program_0"],
                            self.tokenizer.vocab[4]["Position_0"],
                            self.tokenizer.vocab[5]["Bar_0"],
                        ]
                    )
                else:  # CPWord
                    sample.append(
                        [
                            self.tokenizer.vocab[0]["Family_Note"],
                            self.tokenizer.vocab[1]["Ignore_None"],
                            self.pitches[current_pitch_idx],
                            self.velocities[current_vel_idx],
                            self.duration[current_dur_idx],
                        ]
                    )
            sample = LongTensor(sample).unsqueeze(1)  # (N,1,Z)
        else:  # nb_steps mode, (N,1)
            sample = (
                arange(self.step, min([self.step + self.batch_size, self.nb_samples]))
                .unsqueeze(-1)
                .long()
            )
        self.step += self.batch_size
        return sample


if __name__ == "__main__":
    from exp_generation import experiments as exp_gen
    from exp_pretrain import experiments as exp_cla

    from torch import randint
    from transformers.trainer_utils import get_last_checkpoint

    batch_size_ = 20000
    device_ = select_device(True, False)  # CUDA
    data_tok_already_done = []

    row_names = ["No BPE", "BPE 1k", "BPE 5k", "BPE 10k", "BPE 20k", "PVm", "PVDm"]
    id_column_names = [f"{metric if isinstance(metric, str) else metric.__name__} {dataset} {tok}"
                       for metric in ["isoscore"] + id_func
                       for dataset in ["Maestro", "MMD"]
                       for tok in ["TSD", "REMI"]]
    id_isoscore_df = DataFrame(index=row_names, columns=id_column_names)
    column_names = [f"{dataset} {tok}" for metric in id_func
                    for dataset in ["Maestro", "MMD"] for tok in ["TSD", "REMI"]]
    avg_neighbor_var_df = DataFrame(index=row_names, columns=column_names)

    for exp in exp_gen + exp_cla:
        tokenization = exp.name.split('_')[-1]
        data_tok = f"{exp.dataset}_{tokenization}"
        tok = "REMI" if tokenization.startswith("REMI") else "TSD"
        if data_tok in data_tok_already_done:
            continue
        set_seed(777)
        (out_dir := Path("analysis", "embeddings_space", data_tok)).mkdir(
            parents=True, exist_ok=True
        )

        baseline_names = [
            baseline_.name.split("_")[-1]
            for baseline_ in exp.baselines
            if not baseline_.tokenizer.is_multi_voc
        ]
        xticklabels = [
            f"BPE {int(int(name[3:]) / 1000)}k" if name[:3] == "bpe" else
            name[len(tokenization):] if name[:len(tokenization)] == tokenization else
            "No BPE" if name == "noBPE" else
            name
            for name in baseline_names
        ]

        singular_values, intrinsic_dimensions, isoscores, mean_vars = [], [], [], []
        for bi, baseline_ in enumerate(exp.baselines):
            if baseline_.tokenizer.is_multi_voc:
                continue
            print(f"{data_tok} - {baseline_.name}")

            # Load model
            if not exp.name.startswith("gen"):
                pt_path = Path(
                    "runs", "cla_pre_trained", f"{exp.dataset}_{baseline_.name}"
                )
            else:
                pt_path = baseline_.run_path
            last_checkpoint = get_last_checkpoint(pt_path)
            model = baseline_.create_model()
            kwargs = {}
            if baseline_.tokenizer.is_multi_voc:
                embed_pool_size = [
                    baseline_.embed_pooling_size
                    for _ in range(len(baseline_.tokenizer.len))
                ]
                kwargs = {
                    "num_classes": baseline_.tokenizer.len,
                    "embed_sizes": embed_pool_size,
                }
            model = model.from_pretrained(last_checkpoint, **kwargs).to(device_)

            # Create embeddings
            embeddings_ = []
            with no_grad():
                embed_module = model.get_input_embeddings()
                for sample_ in TokenIterator(
                    batch_size_, baseline_.tokenizer, MAX_NB_EMBEDS
                ):
                    emb = embed_module(sample_.to(device_))[:, 0]  # (N,1,E) --> (N,E)
                    embeddings_.append(emb)
            embeddings_ = cat(embeddings_, 0)  # (V,E)
            embeddings_np = embeddings_.cpu().numpy()  # (V,E)

            # Plot cosine sim
            compute_embedding_distances(embeddings_, out_dir / f"cosim_{baseline_.name}.pdf")

            # Mean variance of distances of neighbouring embeddings
            mean_vars.append(compute_mean_var_embeds(embeddings_))
            avg_neighbor_var_df.at[row_names[bi], f"{exp.dataset} {tok}"] = round(mean_vars[-1], 3)

            # Isoscore
            isoscores.append(isoscore(embeddings_np))
            id_isoscore_df.at[row_names[bi], f"isoscore {exp.dataset} {tok}"] = round(isoscores[-1], 3)
            print(f"IsoScore: {isoscores[-1]:.2f}")

            # Compute singular value
            singular_values.append(scipy.linalg.svdvals(embeddings_np.T))

            # Intrinsic dimensions
            id_results = compute_intrinsic_dimensions(embeddings_)
            intrinsic_dimensions.append(id_results)
            column_offset_tok = 1 if exp.name.endswith("REMI") or exp.name.endswith("REMIPlus") else 0
            column_offset_data = 1 if exp.dataset == "MMD" else 0
            for id_res, id_func_ in zip(id_results, id_func):
                id_isoscore_df.at[row_names[bi], f"{id_func_.__name__} {exp.dataset} {tok}"] = round(id_res, 1)

            # Plot UMAP of embeddings
            # Reduce to n points to keep it not to heavy to load as vectorized in paper pdf
            if len(baseline_.tokenizer) > MAX_NB_POINTS_UMAP:
                token_ids_to_keep = randint(
                    0, len(baseline_.tokenizer), (MAX_NB_POINTS_UMAP,)
                ).tolist()
                embeddings_ = embeddings_[token_ids_to_keep]
            else:
                token_ids_to_keep = list(range(len(baseline_.tokenizer)))

            token_types_ = get_token_types(
                baseline_, token_ids_to_keep
            )  # get token types of specific tokens

            reducer = UMAP(n_components=2)
            proj = reducer.fit_transform(embeddings_.cpu())  # (V,2)
            plot_points(
                proj,
                token_types_,
                out_dir / f"umap_2d_{exp.name}_{baseline_.name}.pdf",
            )

            reducer = UMAP(n_components=3)
            proj = reducer.fit_transform(embeddings_.cpu())  # (V,3)
            plot_points_3d(
                proj,
                token_types_,
                out_dir / f"umap_3d_{exp.name}_{baseline_.name}.pdf",
            )

        """# Save metrics as CSV
        with open(
            out_dir / f"intrinsic_dimension.csv", "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Baseline"] + [id_.__name__ for id_ in id_func])
            for baseline_, row in zip(exp.baselines, intrinsic_dimensions):
                writer.writerow([baseline_.name] + row)
        with open(out_dir / f"isoscore.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([baseline_.name for baseline_ in exp.baselines])
            writer.writerow(isoscores)
        with open(
            out_dir / f"mean_var_neighbors.csv", "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow([baseline_.name for baseline_ in exp.baselines])
            writer.writerow(mean_vars)"""

        # Plot singular values
        try:
            model_dim = exp.baselines[0].model_config.hidden_size  # BERT
        except AttributeError:
            model_dim = exp.baselines[0].model_config.n_embd  # GPT2
        old_indices = np.arange(len(singular_values[0]))
        new_indices = np.linspace(0, len(singular_values[0]) - 1, model_dim)
        # adjusted_no_bpe = UnivariateSpline(old_indices, singular_values[0], k=3, s=0)(new_indices)
        # adjusted_no_bpe = adjusted_no_bpe * (1 / adjusted_no_bpe[0])
        for si in range(len(singular_values)):
            singular_values[si] = singular_values[si] * (1 / singular_values[si][0])
        fig_, ax_ = plt.subplots(figsize=(6, 5))
        for bi, (sing_val, baseline_name) in enumerate(
            zip(singular_values, xticklabels)
        ):  # (E)
            ax_.semilogx(
                np.arange(1, sing_val.shape[0] + 1), sing_val, label=baseline_name
            )
        # ax_.semilogx(np.arange(1, model_dim + 1), adjusted_no_bpe, label='noBPE adj.', linestyle='dotted',
        #             color=mcolors.TABLEAU_COLORS['tab:blue'])
        ax_.set_xlim(left=1, right=model_dim + 50)
        ax_.set_xlabel("Dimension", fontsize=17)
        ax_.set_ylabel("Singular value", fontsize=17)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(prop={"size": 16})
        # plt.savefig(out_dir / f"singular_value_{exp.name}.pdf", bbox_inches="tight", dpi=300)
        extent = (
            ax_
            .get_window_extent()
            .transformed(fig_.dpi_scale_trans.inverted())
            .translated(-0.38, -0.3)
            .expanded(1.22, 1.2)
        )
        fig_.savefig(out_dir / f"singular_value_{exp.name}.pdf", bbox_inches=extent, dpi=300)
        fig_.clf()

        # plot intrinsic dim as
        fig_, ax1 = plt.subplots(figsize=(3.5, 3))
        lns = []

        marker_offset = 0
        ax2 = None
        lPCA_score = [s[0] for s in intrinsic_dimensions]
        if max(lPCA_score) >= SECOND_AXIS_LIM:
            ax2 = ax1.twinx()
            vals, ticks = [], []
            for bi in range(len(intrinsic_dimensions)):  # filter non-valid scores
                if intrinsic_dimensions[bi][0] is not None and not np.isnan(
                    intrinsic_dimensions[bi][0]
                ):
                    vals.append(intrinsic_dimensions[bi][0])
                    ticks.append(bi)
                del intrinsic_dimensions[bi][0]
            lns.append(
                ax2.scatter(
                    ticks, vals, label=id_func[0].__name__, marker=id_markers[0]
                )
            )
            del id_func[0]
            marker_offset = 1
        for fi, func_ in enumerate(id_func):
            vals, ticks = [], []
            for bi in range(len(intrinsic_dimensions)):  # filter non-valid scores
                if intrinsic_dimensions[bi][fi] is not None and not np.isnan(
                    intrinsic_dimensions[bi][fi]
                ):
                    vals.append(intrinsic_dimensions[bi][fi])
                    ticks.append(bi)  # dif color and marker per function
            lns.append(
                ax1.scatter(
                    ticks,
                    vals,
                    label=func_.__name__,
                    marker=id_markers[fi + marker_offset],
                    c=list(mcolors.TABLEAU_COLORS.values())[fi + marker_offset],
                )
            )

        ax1.set_xticks(range(len(intrinsic_dimensions)))
        ax1.set_xticklabels(xticklabels, rotation=72)
        ax1.grid("on")
        if (
            exp.baselines[0].tokenization == "TSD"
        ):  # Dimension ylabel on top left figure
            ax1.set_ylabel("Dimension")
        labs = [line.get_label() for line in lns]
        if ax2 is not None:
            ax2.legend(lns, labs, loc=0, prop={"size": 8})
        else:
            ax1.legend(lns, labs, loc=0, prop={"size": 8})
        # plt.legend(prop={'size': 8})
        plt.savefig(
            out_dir / f"intrinsic_dim_{exp.name}.pdf", bbox_inches="tight", dpi=300
        )
        fig_.clf()

        data_tok_already_done.append(data_tok)

    # Saving metrics as csv / latex
    out_dir = Path("analysis", "embeddings_space")

    id_isoscore_df.to_csv(out_dir / "id_isoscore.csv")
    # id_df = highlight_df_latex(id_df)
    id_isoscore_df.to_latex(out_dir / "id_isoscore.txt", bold_rows=True)

    avg_neighbor_var_df.to_csv(out_dir / "avg_neighbor_var.csv")
    # avg_neighbor_var_df = highlight_df_latex(avg_neighbor_var_df)
    avg_neighbor_var_df.to_latex(out_dir / "avg_neighbor_var.txt", bold_rows=True)
