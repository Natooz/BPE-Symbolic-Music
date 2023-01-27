#!/usr/bin/python3 python

"""Plots cosine sim of embeddings of BPE experiments
"""
from pathlib import Path
import logging

import miditok
import numpy as np
import scipy.linalg
from torch import Tensor, LongTensor, cat, load, no_grad, mean, std, var, full, triu, cosine_similarity, zeros, \
    multinomial, arange, abs as abs_pt
from torchtoolkit.train import select_device
from torchtoolkit.utils import seed_everything
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import UnivariateSpline
from umap import UMAP
from skdim.id import lPCA, MLE, MOM, TLE, TwoNN, FisherS
from IsoScore import IsoScore
from tqdm import tqdm

from exp_gen import experiments
from classes import Baseline
from constants import MAX_NB_COMPOSERS


nb_bpe_type_successions_to_plot = 5
tok_typ_short = {'Pitch': 'Pch', 'Velocity': 'Vel', 'Duration': 'Dur', 'Time-Shift': 'TS', 'Position': 'Pos'}
max_nb_embed_pooling = 50000
min_nb_embed_per_token_type_analysis = 30  # don't perform cosim / isotropy / intrinsic dimension without at least this
id_func = [lPCA, MLE, MOM, TLE, TwoNN, FisherS]
markers = ['o', 'v', '^', 'x', 'p', '*']


def analyze_embeddings(baseline: Baseline, embeddings: Tensor, logger: logging.Logger, log_suffix: str,
                       out_path: Path = None, pw_cosim_eucl_dist: bool = False):
    logger.debug(f'\n{baseline.exp_name} - {baseline.name}{"" if log_suffix == "" else f" ({log_suffix})"}')

    if pw_cosim_eucl_dist:
        # Euclidian distance and cosine similarity
        # (https://aclanthology.org/D19-1006/)
        # (A Contrastive Framework for Neural Text Generation)
        # We do not batch compute them as it requires too much memory
        # pw_eucl_dis = cdist(embeddings, embeddings)  # batched
        # pw_cos_sim = cosine_similarity(embeddings[:, :, None], embeddings.t()[None, :, :]).cpu()
        pw_eucl_dis = zeros((embeddings.shape[0], embeddings.shape[0])).to(embeddings.device)
        pw_cos_sim = zeros((embeddings.shape[0], embeddings.shape[0])).to(embeddings.device)
        for i in range(embeddings.shape[0]):  # non batched
            for j in range(embeddings.shape[0]):  # compute full matrix as we will plot cosine sim
                pw_eucl_dis[i, j] = (embeddings[i] - embeddings[j]).pow(2).sum().sqrt().unsqueeze(0)
                pw_cos_sim[i, j] = cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0))
        triu_mask = triu(full(pw_cos_sim.shape, 1).bool(),  diagonal=1)
        pw_eucl_dis_vals = pw_eucl_dis[triu_mask]
        pw_cos_sim_vals = abs_pt(pw_cos_sim[triu_mask])
        mean_pw_dis, std_pw_dis, var_pw_dis = mean(pw_eucl_dis_vals), std(pw_eucl_dis_vals), var(pw_eucl_dis_vals)
        mean_pw_cs, std_pw_cs, var_pw_cs = mean(pw_cos_sim_vals), std(pw_cos_sim_vals), var(pw_cos_sim_vals)
        logger.debug(f'Mean euclidian distance: {mean_pw_dis:.2f} +- {std_pw_dis:.2f} - var: {var_pw_dis:.2f}\n'
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

    # Isoscore
    embed_np = embeddings.cpu().numpy()
    isoscore = IsoScore.IsoScore(embed_np.T)
    logger.debug(f'IsoScore: {isoscore:.2f}')

    # Intrinsic Dimension
    # id_func = [MLE, DANCo, lPCA, MOM, TLE, TwoNN, FisherS]
    scores = [isoscore]
    for func in id_func:
        id_ = func()
        res, id_score = 'error', None
        try:
            id_ = id_.fit(embed_np)
            res, id_score = f'{id_.dimension_:.2f}', id_.dimension_
        except ValueError:
            res = 'Value error (nan or inf)'
        finally:
            logger.debug(f'Intrinsic dimension - {id_.__class__.__name__}: {res}')
            scores.append(id_score)

    return scores


def plot_points(points: np.ndarray, token_types: dict, out_path: Path):
    # Plots by token type
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    for token_type, indices in token_types.items():
        plt.scatter(*points[indices].T, label=token_type, s=4)
        '''if baseline.bpe_factor == 0:  # display value of each points
            for token in indices:
                txt = baseline.tokenizer[token].split('_')[1]
                ax.annotate(txt, t_sne.embedding_[token])'''
    plt.legend().set_zorder(100)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    fig.clf()


def plot_points_3d(points, token_types: dict, out_path: Path):
    # Plots by token type
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for token_type, indices in token_types.items():
        ax.scatter(*points[indices].T, label=token_type)
    # Then BPE by type successions + others
    plt.legend(prop={'size': 8}).set_zorder(100)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    fig.clf()


def get_token_types(baseline: Baseline) -> dict:
    # Handles labels for the legend of t-SNE plot
    # Compute recurrent BPE token type successions, and only display the most
    # recurrent ones, others are showed as "Other"
    if baseline.is_embed_pooling:  # get the basic token types
        token_types = {'All': list(range(max_nb_embed_pooling))}  # not differentiated for embed pooling
    else:
        (token_types := {'Special': []}).update(baseline.tokenizer.vocab._token_types_indexes)
        for typ in ['PAD', 'SOS', 'EOS']:
            token_types['Special'].append(token_types[typ][0])
            del token_types[typ]

    if baseline.bpe_factor > 0:
        bpe_types_successions = {}
        for tok in token_types['BPE']:
            sub_tok_types = list(map(int, baseline.tokenizer[tok].split('_')[1].split('.')[1].split('-')))
            sub_tok_types = [baseline.tokenizer.vocab.token_type(t) for t in sub_tok_types]
            sub_tok_types = '-'.join([(tok_typ_short[t] if t in tok_typ_short else t) for t in sub_tok_types])
            try:
                bpe_types_successions[sub_tok_types] += [tok]
            except KeyError:
                bpe_types_successions[sub_tok_types] = [tok]
        bpe_types_successions = dict(sorted(bpe_types_successions.items(), key=lambda i: -len(i[1])))
        bpe_types_successions['Other BPE'] = []
        for key, tokens in list(bpe_types_successions.items())[nb_bpe_type_successions_to_plot:]:
            if key == 'Other BPE':
                break
            bpe_types_successions['Other BPE'] += tokens
            del bpe_types_successions[key]
        token_types.update(bpe_types_successions)  # update dict with decomposed BPE types
        del token_types['BPE']  # remove composed BPE tokens, don't need it anymore

    return token_types


class TokenIterator:
    def __init__(self, batch_size: int, tokenizer: miditok.MIDITokenizer, embed_pooling_nb_embed: int = None):
        """Training iterator class.
        Can work in two modes:
            1. Number of steps: will be iterated a fixed number of times
            2. Min valid accuracy: will be iterated till the model reaches a target validation
                accuracy value, or if the number of training steps exceeds max_nb_steps.
        :param batch_size: batch size
        :param tokenizer: tokenizer
        :param embed_pooling_nb_embed: maximum nb of token combinations to use for embedding pooling tokenizations
        """
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        if tokenizer.is_multi_voc:
            if isinstance(tokenizer, miditok.OctupleMono):
                self.pitches = tokenizer.vocab[0].tokens_of_type('Pitch')
                self.velocities = tokenizer.vocab[1].tokens_of_type('Velocity')
                self.duration = tokenizer.vocab[2].tokens_of_type('Duration')
            elif isinstance(tokenizer, miditok.CPWord):  # only consider Pitch / Vel / Dur combinations
                self.pitches = tokenizer.vocab[2].tokens_of_type('Pitch')
                self.velocities = tokenizer.vocab[3].tokens_of_type('Velocity')
                self.duration = tokenizer.vocab[4].tokens_of_type('Duration')

            self.nb_samples = len(self.pitches) * len(self.velocities) * len(self.duration)
            if self.nb_samples > embed_pooling_nb_embed:
                self.samples_idx = multinomial(arange(self.nb_samples).float(), embed_pooling_nb_embed).long().sort()
                self.nb_samples = len(self.samples_idx)
            else:
                self.samples_idx = arange(self.nb_samples)
        else:
            self.nb_samples = len(tokenizer)

    def __getitem__(self, idx: int) -> LongTensor:
        if not isinstance(self.tokenizer.vocab, list):
            return LongTensor([idx])
        current_pitch_idx = idx // (len(self.duration) * len(self.velocities))
        current_vel_idx = (idx // len(self.duration)) % len(self.velocities)
        current_dur_idx = idx % len(self.duration)
        if isinstance(self.tokenizer, miditok.OctupleMono):
            token = [self.pitches[current_pitch_idx], self.velocities[current_vel_idx], self.duration[current_dur_idx],
                     self.tokenizer.vocab[3]['Position_0'], self.tokenizer.vocab[4]['Bar_0']]
        else:  # CPWord
            token = [self.tokenizer.vocab[0]['Family_Note'], self.tokenizer.vocab[1]['Position_Ignore'],
                     self.pitches[current_pitch_idx], self.velocities[current_vel_idx], self.duration[current_dur_idx]]

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
            for s in range(self.step, min([self.step + self.batch_size, self.nb_samples])):
                comb_idx = self.samples_idx[s]
                current_pitch_idx = comb_idx // nb_vel_dur_comb
                current_vel_idx = (comb_idx // len(self.duration)) % len(self.velocities)
                current_dur_idx = comb_idx % len(self.duration)
                if isinstance(self.tokenizer, miditok.OctupleMono):
                    sample.append([self.pitches[current_pitch_idx], self.velocities[current_vel_idx],
                                   self.duration[current_dur_idx], self.tokenizer.vocab[3]['Position_0'],
                                   self.tokenizer.vocab[4]['Bar_0']])
                else:  # CPWord
                    sample.append([self.tokenizer.vocab[0]['Family_Note'], self.tokenizer.vocab[1]['Position_Ignore'],
                                   self.pitches[current_pitch_idx], self.velocities[current_vel_idx],
                                   self.duration[current_dur_idx]])
            sample = LongTensor(sample).unsqueeze(1)  # (N,1,Z)
        else:  # nb_steps mode, (N,1)
            sample = arange(self.step, min([self.step + self.batch_size, self.nb_samples])).unsqueeze(-1).long()
        self.step += self.batch_size
        return sample


if __name__ == '__main__':
    # from comp_classification import cla_model_conf_large
    batch_size_ = 20000
    device = select_device(True)  # CUDA

    # Loads tokenizers
    for exp in experiments:
        for baseline_ in exp.baselines:
            baseline_.load_tokenizer()

    for model_typ in ['gen', 'cla_pt']:
        seed_everything(777)
        (out_dir := Path('analysis', 'embeddings_space', model_typ)).mkdir(parents=True, exist_ok=True)
        logger_ = logging.getLogger('embeddings')
        logger_.handlers = []
        logger_.addHandler(logging.FileHandler(out_dir / 'embeddings_space.log'))
        logger_.addHandler(logging.StreamHandler())
        logger_.setLevel(logging.DEBUG)

        isoscores = []

        for exp in tqdm(experiments):
            # exp.cla_model_conf = cla_model_conf_large
            if (model_typ == 'cla_pt' and exp.dataset != 'GiantMIDI') or \
                    (model_typ == 'gen' and exp.name[-5:] == 'LARGE'):
                continue  # classification only on GiantMIDI dataset
            singular_values, isotropy_scores, isoscores_exp = [], [], []
            for baseline_ in exp.baselines:
                # Load model
                if model_typ == 'gen':
                    (model := exp.create_gen(baseline_).to(device)).eval()
                    model.load_state_dict(load(baseline_.run_path / 'checkpoint.pt.tar',
                                               map_location=device)['model_state_dict'])
                else:
                    (model := exp.create_classifier(baseline_, num_labels=MAX_NB_COMPOSERS).to(device)).eval()
                    model.load_state_dict(load(baseline_.run_path_classifier / 'pre_trained' / 'checkpoint.pt.tar',
                                               map_location=device)['model_state_dict'])

                # Create embeddings
                embeddings_ = []
                with no_grad():
                    for sample_ in TokenIterator(batch_size_, baseline_.tokenizer, max_nb_embed_pooling):
                        if model_typ == 'gen':
                            emb = model.transformer.wte(sample_.to(device))[:, 0]  # (N,1,E) --> (N,E)
                        else:
                            emb = model.bert.embeddings.word_embeddings(sample_.to(device))[:, 0]  # (N,1,E) --> (N,E)
                        embeddings_.append(emb)
                embeddings_ = cat(embeddings_, 0)  # (V,E)
                embeddings_np = embeddings_.cpu().numpy()  # (V,E)

                # Compute singular value
                singular_values.append(scipy.linalg.svdvals(embeddings_np.T))

                # Get token types and associated tokens
                token_types_ = get_token_types(baseline_)

                # Analyze embeddings + plot cosine sim matrix
                # pw_cosim = True if baseline_.tokenization in ['REMI', 'TSD'] and baseline_.bpe_factor <= 20 else False
                pw_cosim = False
                scores_ = analyze_embeddings(baseline_, embeddings_, logger_, '', pw_cosim_eucl_dist=pw_cosim,
                                             out_path=out_dir / f'pw_cos_sim_{exp.name}_{baseline_.name}.pdf')
                isoscores_exp.append(scores_[0])
                isotropy_scores.append(scores_[1:])  # (B,F)
                '''for token_type_, indices_ in token_types_.items():
                    if len(indices_) >= min_nb_embed_per_token_type_analysis:
                        analyze_embeddings(baseline_, embeddings_[indices_], logger_, token_type_,
                                           out_dir / exp.name / 'pw_cosine_sim' / token_type_ / f'{baseline_.name}.pdf')
                                           '''

                # Plot T-SNE and UMAP
                if not baseline_.is_embed_pooling:
                    """(t_sne := TSNE(n_iter=4000, init='pca', learning_rate=200.0)).fit(embeddings_np)  # (V,2)
                    plot_points(t_sne.embedding_, token_types_,
                                out_dir / exp.name / 't_sne_2d' / f't_sne_2d_{exp.name}_{baseline_.name}.png')
                    (t_sne := TSNE(3, n_iter=4000, init='pca', learning_rate=200.0)).fit(embeddings_)  # (V,3)
                    plot_points_3d(t_sne.embedding_, token_types_,
                                   out_dir / exp.name / 't_sne_3d' / f't_sne_3d_{exp.name}_{baseline_.name}.png')"""

                    reducer = UMAP(n_components=2)
                    proj = reducer.fit_transform(embeddings_np)  # (V,2)
                    plot_points(proj, token_types_,
                                out_dir / exp.name / 'umap_2d' / f'umap_2d_{exp.name}_{baseline_.name}.png')
                    reducer = UMAP(n_components=3)
                    proj = reducer.fit_transform(embeddings_)  # (V,3)
                    plot_points_3d(proj, token_types_,
                                   out_dir / exp.name / 'umap_3d' / f'umap_3d_{exp.name}_{baseline_.name}.png')

            isoscores.append(isoscores_exp)

            # Plot singular values
            model_dim = exp.cla_model_conf.dim if model_typ == 'cla_pt' else exp.baselines[0].model_config.dim
            old_indices = np.arange(len(singular_values[0]))
            new_indices = np.linspace(0, len(singular_values[0]) - 1, model_dim)
            adjusted_no_bpe = UnivariateSpline(old_indices, singular_values[0], k=3, s=0)(new_indices)
            adjusted_no_bpe = adjusted_no_bpe * (1 / adjusted_no_bpe[0])
            for si in range(len(singular_values)):
                singular_values[si] = singular_values[si] * (1 / singular_values[si][0])
            fig_, ax_ = plt.subplots(figsize=(6, 5))
            for bi, (sing_val, baseline_) in enumerate(zip(singular_values, exp.baselines)):  # (E)
                ax_.semilogx(np.arange(1, sing_val.shape[0] + 1), sing_val,
                             label=baseline_.name if baseline_.name != 'OctupleMono' else 'Octuple')
            ax_.semilogx(np.arange(1, model_dim + 1), adjusted_no_bpe, label='noBPE adj.', linestyle='dotted',
                         color=mcolors.TABLEAU_COLORS['tab:blue'])
            ax_.set_xlim(left=1, right=model_dim+50)
            ax_.set_xlabel('Dimension')
            ax_.set_ylabel('Singular value')
            plt.legend(prop={'size': 11})
            plt.savefig(out_dir / f'singular_value_{exp.name}.pdf', bbox_inches='tight', dpi=300)
            fig_.clf()

            # plot intrinsic dim as
            fig_, ax1 = plt.subplots(figsize=(3.5, 3))
            lns = []
            if model_typ == 'cla_pt':
                ax2 = ax1.twinx()
                vals, ticks = [], []
                for bi in range(len(isotropy_scores)):  # filter non-valid scores
                    if isotropy_scores[bi][0] is not None and not np.isnan(isotropy_scores[bi][0]):
                        vals.append(isotropy_scores[bi][0])
                        ticks.append(bi)
                lns.append(ax2.scatter(ticks, vals, label=id_func[0].__name__, marker=markers[0]))
            for fi, func_ in enumerate(id_func):
                if model_typ == 'cla_pt' and fi == 0:
                    continue
                vals, ticks = [], []
                for bi in range(len(isotropy_scores)):  # filter non-valid scores
                    if isotropy_scores[bi][fi] is not None and not np.isnan(isotropy_scores[bi][fi]):
                        vals.append(isotropy_scores[bi][fi])
                        ticks.append(bi)  # dif color and marker per function
                lns.append(ax1.scatter(ticks, vals, label=func_.__name__, marker=markers[fi],
                                       c=list(mcolors.TABLEAU_COLORS.values())[fi]))

            ax1.set_xticks(range(len(isotropy_scores)))
            ax1.set_xticklabels([baseline_.name if baseline_.name != 'OctupleMono' else 'Octuple'
                                 for baseline_ in exp.baselines], rotation=72)
            ax1.grid('on')
            if exp.baselines[0].tokenization == 'TSD' and (exp.dataset == 'POP909-merged' or
                                                           (model_typ == 'cla_pt' and exp.name[-5:] != 'LARGE')):
                ax1.set_ylabel('Dimension')
            labs = [line.get_label() for line in lns]
            if model_typ == 'cla_pt':
                ax2.legend(lns, labs, loc=0, prop={'size': 8})
            else:
                ax1.legend(lns, labs, loc=0, prop={'size': 8})
            # plt.legend(prop={'size': 8})
            plt.savefig(out_dir / f'intrinsic_dim_{exp.name}.pdf', bbox_inches='tight', dpi=300)
            fig_.clf()

        # plot isoscore
        fig_, ax_ = plt.subplots(figsize=(6, 4))
        for fi, isoscores_exp in enumerate(isoscores):
            ax_.scatter(list(range(len(isoscores_exp))), isoscores_exp,
                        label=f'{experiments[fi].dataset} {experiments[fi].baselines[0].tokenization}',
                        marker=markers[fi])
        ax_.set_xticks(range(len(isoscores[1])))
        ax_.set_xticklabels([baseline_.name if baseline_.name != 'OctupleMono' else 'Octuple'
                             for baseline_ in experiments[1].baselines], rotation=75)
        ax_.grid('on')
        plt.legend()
        plt.savefig(out_dir / f'isoscore_{model_typ}.pdf', bbox_inches='tight', dpi=300)
        fig_.clf()
