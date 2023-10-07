#!/usr/bin/python3 python

"""Plots cosine sim of embeddings of BPE experiments

"""
from pathlib import Path
import logging

from torch import Tensor, load, no_grad, histogram, where, arange, topk
from transformers.trainer_utils import set_seed

from experiments import experiments
from scripts.bpe_embedding_space import compute_mean_var_embeds
from scripts.utils import get_token_dist
from utils import select_device


nb_bpe_type_successions_to_plot = 5
tok_typ_short = {
    "Pitch": "Pch",
    "Velocity": "Vel",
    "Duration": "Dur",
    "TimeShift": "TS",
    "Position": "Pos",
}
max_nb_embed_pooling = 50000
min_nb_embed_per_token_type_analysis = (
    30  # don't perform cosim / isotropy / intrinsic dimension without at least this
)
ratio_most_sampled = 0.5


if __name__ == "__main__":
    batch_size_ = 20000
    device = select_device(True)  # CUDA

    # Loads tokenizers
    for exp in experiments:
        for baseline_ in exp.baselines:
            baseline_.load_tokenizer()

    set_seed(777)
    (out_dir := Path("analysis")).mkdir(parents=True, exist_ok=True)
    logger_ = logging.getLogger("embeddings")
    logger_.addHandler(logging.FileHandler(out_dir / "isotropy.log"))
    logger_.addHandler(logging.StreamHandler())
    logger_.setLevel(logging.DEBUG)

    for exp in experiments:
        for baseline_ in exp.baselines:
            # Load model
            (model := exp.create_gen(baseline_).to(device)).eval()
            model.load_state_dict(
                load(baseline_.run_path / "checkpoint.pt.tar", map_location=device)[
                    "model_state_dict"
                ]
            )
            generated_files_paths = list(baseline_.gen_data_path.glob("**/*.json"))

            # Get sampled tokens dist
            tokens_dist = Tensor(get_token_dist(generated_files_paths))  # (V)
            tokens_hist = histogram(
                tokens_dist, arange(0, len(baseline_.tokenizer)).float()
            ).hist
            tokens_sampled = where(tokens_hist != 0)[0]
            tokens_most_sampled = topk(
                tokens_sampled, k=int(len(tokens_sampled) * ratio_most_sampled)
            ).indices

            # Create and analyze embeddings
            with no_grad():
                embeddings = model.transformer.wte(
                    tokens_sampled.unsqueeze(0).to(device)
                )[
                    0
                ]  # (1,V',E) --> (V',E)
                embeddings_most_sampled = model.transformer.wte(
                    tokens_most_sampled.unsqueeze(0).to(device)
                )[0]
            compute_mean_var_embeds(
                baseline_,
                embeddings,
                logger_,
                f"only tokens sampled ({len(tokens_sampled)}/{len(baseline_.tokenizer)})",
                plot_embed_cosim_mat=False,
            )
            compute_mean_var_embeds(
                baseline_,
                embeddings_most_sampled,
                logger_,
                f"{ratio_most_sampled*100}% most sampled ({len(tokens_most_sampled)})",
                plot_embed_cosim_mat=False,
            )
