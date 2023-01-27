
from typing import List, Union, Any
from random import shuffle
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm


def mean_std(dist: Union[List[Any], np.ndarray]) -> str:
    if isinstance(dist, list):
        dist = np.array(dist)
    return f'{np.mean(dist):.2f} ± {np.std(dist):.2f}'


def get_token_dist(files_paths: List[Path]) -> np.ndarray:
    # Retrieve token dists
    tokens_dist = []
    for file_path in tqdm(files_paths, 'loading files'):
        with open(file_path) as json_file:
            tokens = json.load(json_file)['tokens'][0]  # first track only, the generated one for gen files
        tokens_dist += tokens

    return np.array(tokens_dist)


def reduce_list(list_: List, limit: int) -> List:
    if len(list_) > limit:
        shuffle(list_)
        list_ = list_[:limit]
    return list_
