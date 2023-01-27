#!/usr/bin/python3 python

"""
Composer classification
"""

from pathlib import Path
import logging
import json

from torch import Tensor, load, mean, std, var
from torch.utils.data import DataLoader
from torchtoolkit.train import select_device
from torchtoolkit.utils import seed_everything
from torchtoolkit.data import create_subsets
from torchtoolkit.metrics import calculate_accuracy
from tqdm import tqdm

from classes import Experiment
from dataset import MIDIDatasetClassification
from exp_gen import experiments, cla_model_conf_large
from training import train_classifier
from constants import RANDOM_RATIO_RANGE, MAX_NB_COMPOSERS, CLA_LARGE_BATCH_SIZE


dataset_ = 'GiantMIDI'


def train_all_classifiers(exp: Experiment):
    r"""Trains the generative models (baselines) for a given experiment

    :param exp: experiment associated
    :return: discriminator model trained
    """
    for baseline in exp.baselines:
        pt_path = baseline.run_path_classifier / 'pre_trained'
        ft_path = baseline.run_path_classifier / 'fine_tuned'
        if Path(baseline.run_path_classifier, 'all_results.json').exists():
            continue  # already trained and tested
        seed_everything(baseline.seed)
        baseline.run_path_classifier.mkdir(parents=True, exist_ok=True)
        device = select_device(exp.gen_train_conf.use_cuda)
        logger = logging.getLogger(exp.name)
        (fh := logging.FileHandler(baseline.run_path_classifier / 'training.log')).setLevel(logging.DEBUG)
        (sh := logging.StreamHandler()).setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.setLevel(logging.DEBUG)

        # Loads data
        baseline.load_tokenizer()
        random_range = [3, len(baseline.tokenizer)] if not baseline.is_embed_pooling \
            else [[3, len(v)] for v in baseline.tokenizer.vocab]
        dataset = MIDIDatasetClassification(baseline.data_path, exp.gen_train_conf.min_seq_len,
                                            exp.gen_train_conf.max_seq_len, MAX_NB_COMPOSERS, baseline.pad_token,
                                            baseline.sos_token, baseline.eos_token,
                                            random_range=random_range,
                                            random_ratio_range=RANDOM_RATIO_RANGE)
        subset_train, subset_valid, subset_test = create_subsets(dataset, [exp.gen_train_conf.valid_split,
                                                                           exp.gen_train_conf.test_split])
        dataloader_train = DataLoader(subset_train, exp.gen_train_conf.batch_size, collate_fn=dataset.collate_pre_train)
        dataloader_valid = DataLoader(subset_valid, exp.gen_train_conf.batch_size, collate_fn=dataset.collate_pre_train)

        # Creates the model
        model = exp.create_classifier(baseline, num_labels=dataset.nb_composers, pre_train=True).to(device)

        # Pretrain model
        if not (pt_path / 'checkpoint.pt.tar').is_file():
            train_classifier(exp.cla_pre_train_conf, model, dataloader_train, dataloader_valid, baseline.seed, pt_path,
                             f'PRE-TRAINING {baseline.name}', baseline.pad_token)
        else:
            model.load_state_dict(load(pt_path / 'checkpoint.pt.tar', map_location=device)['model_state_dict'])

        # Fine-tune model
        model.pre_train = False
        if not (ft_path / 'checkpoint.pt.tar').is_file():
            dataloader_train.collate_fn = dataset.collate_fn
            dataloader_valid.collate_fn = dataset.collate_fn
            train_classifier(exp.cla_train_conf, model, dataloader_train, dataloader_valid, baseline.seed, ft_path,
                             f'FINE-TUNING {baseline.name}', baseline.pad_token)

        # Assess model of test subset
        seed_everything(baseline.seed)
        model.load_state_dict(load(ft_path / 'checkpoint.pt.tar', map_location=device)['model_state_dict'])
        model.eval()
        dataloader_test = DataLoader(subset_test, exp.gen_test_conf.batch_size,
                                     collate_fn=dataset.collate_fn_infer)
        all_res = []
        for batch, labels in tqdm(dataloader_test, desc='Testing model'):  # (N,T) or (N,T,Z)
            logits = model.infer(batch.to(device)).cpu()  # (N,T,C)
            all_res.append(calculate_accuracy(logits, labels, mode='softmax'))
        with open(Path(baseline.run_path_classifier, 'all_results.json'), 'w') as f:
            json.dump(all_res, f)
        all_res = Tensor(all_res)
        logger.debug(f'{baseline.name} res: {mean(all_res):.3f} ± {std(all_res):.3f} - var: {var(all_res):.3f}')


if __name__ == '__main__':

    for exp_ in experiments:
        if exp_.dataset != dataset_:
            continue
        for baseline_ in exp_.baselines:
            baseline_.load_tokenizer()

    # Small
    for exp_ in experiments:
        if exp_.dataset != dataset_:
            continue
        """# To run LARGE models, uncomment these line and 
        # modify Baseline.runs_classifier_path to add "_LARGE" at the end
        exp_.cla_model_conf = cla_model_conf_large
        exp_.cla_pre_train_conf.batch_size = CLA_LARGE_BATCH_SIZE
        exp_.cla_train_conf.batch_size = CLA_LARGE_BATCH_SIZE"""
        train_all_classifiers(exp_)
