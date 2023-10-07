#!/usr/bin/python3 python

"""Measure the similarity of the embeddings of the top-k predicted tokens

"""
from pathlib import Path
import logging

from torch import cat, no_grad, mean, std, var, load, topk, triu, full, cdist
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from torchtoolkit.train import select_device
from torchmetrics.functional.text import perplexity
from tqdm import tqdm

from experiments import experiments
from dataset import MIDIDatasetGen


if __name__ == "__main__":
    (out_dir := Path("analysis")).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("embeddings")
    logger.addHandler(logging.FileHandler(out_dir / "embeddings_prediction_sim.log"))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    batch_size = 24
    starting_time_idx = 250  # batch is padded on the left
    top_k = 8
    nb_inferences = 500
    device = select_device(True)  # CUDA

    # Loads tokenizers
    for exp in experiments:
        for baseline in exp.baselines:
            baseline.load_tokenizer()

    for exp in experiments:
        for baseline in exp.baselines:
            if baseline.is_embed_pooling:
                continue  # no embed pooling

            model = exp.create_gen(baseline).to(device)
            model.load_state_dict(
                load(baseline.run_path / "checkpoint.pt.tar", map_location=device)[
                    "model_state_dict"
                ]
            )
            model.eval()
            dataset = MIDIDatasetGen(
                baseline.data_path,
                exp.gen_train_conf.min_seq_len,
                exp.gen_train_conf.max_seq_len,
                baseline.pad_token,
                baseline.bos_token,
            )
            dataloader = DataLoader(
                dataset, batch_size, collate_fn=dataset.collate_fn_infer
            )

            # Create embeddings and computes pairwise cosine similarities
            all_sims = []
            all_dist = []
            all_pplx = []
            with no_grad():
                data = iter(dataloader)
                for i in tqdm(
                    range(nb_inferences),
                    desc=f"{exp.name} {baseline.name} - Testing model",
                ):  # (N,T)
                    try:
                        batch = next(data).to(device)
                    except StopIteration:
                        break
                    attention_mask = (
                        batch != baseline.pad_token
                    ).float()  # 1 for non-masked positions
                    logits = model(batch, attention_mask=attention_mask).logits[
                        :, starting_time_idx:
                    ]  # (N,T,C)
                    top_k_tokens = topk(logits, top_k).indices  # (N,T,k)
                    all_pplx.append(
                        perplexity(
                            logits[:, :-1],
                            batch[:, starting_time_idx + 1 :],
                            baseline.pad_token,
                        ).unsqueeze(0)
                    )

                    # Compute cosine sim between embeddings
                    for seq in top_k_tokens:
                        embeddings = model.transformer.wte(seq)  # (T,k,E)
                        for pos in embeddings:  # (k,E) --> (k,k) --> (k²/2 - k)
                            pairwise_cosine_sim = cosine_similarity(
                                pos[:, :, None], pos.t()[None, :, :]
                            ).cpu()
                            pairwise_eucl_dis = cdist(pos, pos)
                            triu_mask = triu(
                                full(pairwise_cosine_sim.shape, 1).bool(), diagonal=1
                            )
                            all_sims.append(pairwise_cosine_sim[triu_mask])
                            all_dist.append(pairwise_eucl_dis[triu_mask])

            all_sims = cat(all_sims)
            all_dist = cat(all_dist)
            all_pplx = cat(all_pplx)

            # Logs mean similarity
            logger.debug(
                f"{exp.name} - {baseline.name}\n"
                f"PERPLEXITY: {mean(all_pplx):.3f} +- {std(all_pplx):.3f} - var={var(all_pplx):.3f}\n"
                f"PW COSINE SIM: {mean(all_sims):.3f} +- {std(all_sims):.3f} - var={var(all_sims):.3f}\n"
                f"PW EUCLIDIAN DISTANCE {mean(all_dist):.3f} +- {std(all_dist):.3f} - var={var(all_dist):.3f}"
            )
