#!/usr/bin/python3 python

""" Script to sort the MMD dataset
It deduplicate MIDIs by selecting the best versions for each matched audio music.
And make sure it pass our requirements.

"""

import json
from pathlib import PurePath

from miditoolkit import MidiFile
import networkx as nx
import numpy as np
import scipy.optimize
from tqdm import tqdm

from constants import MMD_MIDI_CATS_TO_MERGE, MMD_VALID_PROGRAMS
from scripts.tokenize_datasets import is_midi_valid_multi


def clean_dataset(data_path: str, matches_file_path: str):
    """Cleans the LMD or MMD dataset in order to:
    1. Remove non-valid MIDIs (see is_valid_midi method)
    2. Keep 1 MIDI per 1 audio match:
        the match scores between MIDIs and audios can be viewed as a big
        non-connected bipartite weighted graph on which we will compute a
        matching to keep only pairs of distinct MIDIs and audios
    The selected MIDIs will be saved in a json file to be used to tokenize them

    :param data_path: path containing the data to load, ex: 'data/death_metal_dataset'
    :param matches_file_path: path to the MIDI - audio matches file
    """

    # Reads the MIDI-audio matches file and keeps valid MIDIS for matching
    b = nx.Graph()
    midis = []  # stores all midi filenames (md5s) for validation below
    with open(matches_file_path) as matches_file:
        matches_file.seek(0)
        next(matches_file)  # first line skipped
        for idl, line in enumerate(
            tqdm(matches_file, desc="Reading MMD match file / building the graph")
        ):
            midi_md5, score, audio_sid = line.split()
            midis.append(midi_md5)
            b.add_node(midi_md5, bipartite=0)
            b.add_node(audio_sid, bipartite=1)
            b.add_edge(midi_md5, audio_sid, weight=1 - float(score))

    # Removing invalid MIDIS
    midis = list(set(midis))
    for m, midi_md5 in enumerate(tqdm(midis, desc="Checking MIDIs are valid")):
        try:
            midi = MidiFile(PurePath(data_path, midi_id_to_path(midi_md5)))
            if not is_midi_valid_multi(
                midi,
                3,
                cats_to_merge=MMD_MIDI_CATS_TO_MERGE,
                valid_programs=MMD_VALID_PROGRAMS,
            ):
                b.remove_node(midi_md5)
        except (
            Exception
        ) as _:  # ValueError, OSError, FileNotFoundError, IOError, EOFError, mido.KeySignatureError
            b.remove_node(midi_md5)
    b.remove_nodes_from(list(nx.isolates(b)))

    # Computes matchings
    sub_graphs = [
        b.subgraph(nodes) for nodes in list(nx.connected_components(b))
    ]  # connected bipartite sub-graphs
    matchings = []
    for s, sub_graph in enumerate(
        tqdm(sub_graphs, desc="Computing distinct MIDI-audio matches")
    ):
        matchings.append(match(sub_graph))

    # Sorts matchings
    midi_to_audio = {}
    for m, matching in enumerate(tqdm(matchings, desc="Sorting matchings")):
        for key, value in matching.items():
            midi_md5, audio_sid = (key, value) if len(key) > 22 else (value, key)
            midi_to_audio[midi_md5] = audio_sid

    # Saves the matching file + conversion params, in txt format
    with open(
        PurePath("data", "MMD_METADATA", "midi_audio_matches.json"), "w"
    ) as outfile:
        json.dump(midi_to_audio, outfile, indent=2)


def midi_id_to_path(midi_md5: str) -> PurePath:
    """Returns the relative path of a MIDI file from its file name (md5).

    :param midi_md5: MIDI file name (md5)
    :return: the relative path
    """
    return PurePath(midi_md5[0], midi_md5[1], midi_md5[2], midi_md5).with_suffix(".mid")


def match(graph: nx.Graph) -> dict:
    """Taken and fixed from nx.algorithms.bipartite.minimum_weight_full_matching

    :param graph: connected bipartite graph
    :return: the matching
    """
    if len(graph) == 2:
        return {list(graph.nodes)[0]: list(graph.nodes)[1]}
    left, right = nx.bipartite.sets(graph, None)
    u = list(left)
    v = list(right)
    weights_sparse = nx.algorithms.bipartite.matrix.biadjacency_matrix(
        graph, row_order=u, column_order=v, weight="weight", format="coo"
    )
    weights = np.full(
        weights_sparse.shape, 20e20
    )  # if this raises error, replace np.inf with a very large number
    weights[weights_sparse.row, weights_sparse.col] = weights_sparse.data
    left_matches = scipy.optimize.linear_sum_assignment(weights)
    return {u[ui]: v[vi] for ui, vi in zip(*left_matches)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MetaMIDI Dataset cleaning")
    parser.add_argument(
        "--mmd-path",
        type=str,
        default="data/MMD",
        help="the dataset to run. Corresponds to a path to a file",
    )
    parser.add_argument(
        "--mmd-matches-path",
        type=str,
        default="data/MMD_METADATA/MMD_audio_text_matches.tsv",
        help="MIDI-audio matches file path for MMD"
        "(default: data/MMD_METADATA/MMD_audio_text_matches.tsv)",
    )
    args = parser.parse_args()

    # Cleans MMD
    clean_dataset(args.mmd_path, args.mmd_matches_path)
