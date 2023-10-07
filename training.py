"""
Training functions
"""

from typing import List, Any
from pathlib import Path

from torch import Tensor, randperm, cumsum, argmax, device
from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available
from torch.utils.data import Dataset, Subset, random_split
from transformers import Trainer


def split_object(obj_, split_ratio: List[float]) -> List[Any]:
    r"""Create subsets of a dataset following split ratios.
    if sum(split_ratio) != 1, the remaining portion will be inserted as the first subset

    :param obj_: Object to split, must support indexing and implement the __len__ magic method.
    :param split_ratio: split ratios as a list of float
    :return: the list of subsets
    """
    assert all(0 <= ratio <= 1. for ratio in split_ratio), 'The split ratios must be comprise within [0,1]'
    assert sum(split_ratio) <= 1., 'The sum of split ratios must be inferior or equal to 1'
    nb_samples = len(obj_)
    len_subsets = [int(nb_samples * ratio) for ratio in split_ratio]
    if sum(split_ratio) != 1.:
        len_subsets.insert(0, nb_samples - sum(len_subsets))
    cum_sums = cumsum(Tensor(len_subsets), 0).long()
    idx = randperm(nb_samples)
    idx = [idx[offset - length: offset] for offset, length in zip(cum_sums, len_subsets)]
    split = [[obj_[idx__] for idx__ in idx_] for idx_ in idx]
    return split


def create_subsets(dataset: Dataset, split_ratio: List[float]) -> List[Subset]:
    r"""Create subsets of a dataset following split ratios.
    if sum(split_ratio) != 1, the remaining portion will be inserted as the first subset

    :param dataset: Dataset object, must implement the __len__ magic method.
    :param split_ratio: split ratios as a list of float
    :return: the list of subsets
    """
    assert all(0 <= ratio <= 1. for ratio in split_ratio), 'The split ratios must be comprise within [0,1]'
    assert sum(split_ratio) <= 1., 'The sum of split ratios must be inferior or equal to 1'
    len_subsets = [int(len(dataset) * ratio) for ratio in split_ratio]
    if sum(split_ratio) != 1.:
        len_subsets.insert(0, len(dataset) - sum(len_subsets))
    subsets = random_split(dataset, len_subsets)
    return subsets


def select_device(use_cuda: bool = True, use_mps: bool = True, log: bool = False) -> device:
    r"""Select the device on which PyTorch will run

    :param use_cuda: will run on nvidia GPU if available. (default: True)
    :param use_mps: will run on MPS device if available. (default: True)
    :param log: will log a warning message if a CUDA device is detected but not used. (default: False)
    :return: 'cpu' or 'cuda:0' device object.
    """
    if cuda_available():
        if use_cuda:
            return device("cuda:0")
        elif log:
            print("WARNING: You have a CUDA device, you should probably run with it")
    if mps_available():
        if use_mps:
            return device("mps")
        elif log:
            print("WARNING: You have a MPS device, you should probably run with it")
    return device('cpu')


def is_training_done(run_path: Path) -> bool:
    """Tells if a model has already been trained in the run_path directory,

    :param run_path: model training directory
    :return: if model has already been fully trained
    """
    if run_path.exists():
        if (run_path / "train_results.json").is_file():
            return True

    return False


def is_testing_done(run_path: Path) -> bool:
    """Tells if a model has already been trained in the run_path directory,

    :param run_path: model training directory
    :return: if model has already been fully trained
    """
    if run_path.exists():
        if (run_path / "test_results.json").is_file():
            return True

    return False


def preprocess_logits(logits: Tensor, _: Tensor = None) -> Tensor:
    """Preprocesses the logits before accumulating them during evaluation.
    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids


def train_model(trainer: Trainer):
    r"""Trains a model

    :param trainer: initialized Trainer
    """
    train_result = trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


def test_model(trainer: Trainer, dataset_test: Dataset = None):
    r"""Tests a model, given a baseline

    :param trainer: initialized Trainer
    :param dataset_test: dataset for test / inference data.
    """
    test_results = trainer.predict(dataset_test)
    trainer.log_metrics("test", test_results.metrics)
    trainer.save_metrics("test", test_results.metrics)
