#!/usr/bin/python3 python

"""Parses features and metrics from datasets, and plots figures

"""
from pathlib import Path
import json
from random import shuffle
import logging

from matplotlib import pyplot as plt
from miditok import TSD
from miditoolkit import MidiFile
import seaborn as sns
from tqdm import tqdm
from torchtoolkit.utils import seed_everything
import numpy as np

from exp_gen import experiments, datasets
from tokenize_datasets import is_midi_valid
from constants import PITCH_RANGE, TOKENIZER_PARAMS
from metrics import Consistency, OnsetInterval, NoteDensity, NoteDensityEvolution


if __name__ == '__main__':
    seed_everything(777)
    (out_dir := Path('analysis', 'datasets_features')).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('datasets_features')
    logger.addHandler(logging.FileHandler(out_dir / 'datasets_features.log'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    # Loads tokenizers
    for exp in experiments:
        for baseline in exp.baselines:
            baseline.load_tokenizer()

    metrics = [Consistency(), OnsetInterval(), NoteDensity(), NoteDensityEvolution()]
    tokenizer = TSD(**TOKENIZER_PARAMS)
    file_nb_limit = 1000

    pitches = {dataset: [] for dataset in datasets}
    durations = {dataset: [] for dataset in datasets}
    velocities = {dataset: [] for dataset in datasets}
    onsets = {dataset: [] for dataset in datasets}
    onset_intervals = {dataset: [] for dataset in datasets}

    for di, dataset in enumerate(datasets):
        logger.debug(dataset)
        files_paths = list(Path('data', dataset).glob('**/*.mid'))
        if len(files_paths) > file_nb_limit:
            shuffle(files_paths)
            files_paths = files_paths[:file_nb_limit]

        for f, file_path in enumerate(tqdm(files_paths, desc=f'Parsing {dataset}')):
            try:
                midi = MidiFile(file_path)
            except:
                continue
            if not is_midi_valid(midi, four_beats_per_bar_only=False):
                continue

            ticks_per_bar = 4 * midi.ticks_per_beat
            for track in midi.instruments:
                for metric in metrics[1:]:
                    metric(track.notes)
                for n, note in enumerate(track.notes):
                    pitches[dataset].append(note.pitch)
                    durations[dataset].append((note.end - note.start) / midi.ticks_per_beat)  # in beats
                    velocities[dataset].append(note.velocity)
                    onsets[dataset].append(note.start % (midi.ticks_per_beat * 4) / (midi.ticks_per_beat * 4))
                    if n >= len(track.notes) - 1:
                        continue  # is the last note
                    onset_inter = (track.notes[n + 1].start - track.notes[n].start) / midi.ticks_per_beat
                    if 0 < onset_inter <= 4:  # notes played together or too far apart are discarded
                        onset_intervals[dataset].append(onset_inter)

            # Consistency, done after preprocess
            tokenizer.preprocess_midi(midi)
            for track in midi.instruments:
                metrics[0](track.notes)

        # Logs metrics analysis
        for metric in metrics:
            metric.analyze(logger)
            metric.reset()
        logger.debug('\n')  # line break for better readability

    # Saves the distributions
    with open(out_dir / 'datasets_features_dist.json', 'w') as outfile:
        json.dump({'pitches': pitches, 'durations': durations, 'velocities': velocities,
                   'onsets': onsets, 'onset_intervals': onset_intervals}, outfile)
    '''with open(out_dir / 'datasets_features_dist.json') as file:
        dists = json.load(file)
        pitches = dists['pitches']
        durations = dists['durations']
        velocities = dists['velocities']
        onsets = dists['onsets']
        onset_intervals = dists['onset_intervals']'''

    # plot distributions
    for feature, feature_name in [(pitches, 'pitch'), (durations, 'duration'), (velocities, 'velocity'),
                                  (onsets, 'onset'), (onset_intervals, 'onset_interval')]:
        plt.figure(figsize=(5, 3))
        logger.debug(f'\n{feature_name}')
        for dataset in datasets:
            logger.debug(f'{dataset}: {np.mean(feature[dataset]):.2f} ± {np.std(feature[dataset]):.2f}')
            sns.kdeplot(feature[dataset], label=dataset if dataset != 'POP909-merged' else 'POP909', fill=True)

        # formatting
        plt.legend(title='Dataset', loc='upper left' if feature_name == 'velocity' else 'upper right')  # fontsize=10
        plt.xlabel(feature_name)
        plt.ylabel('density')
        # plt.title(f'{feature} distributions')
        # plt.xlabel(feature)

        if feature_name == 'pitch':
            plt.xlim(PITCH_RANGE.start, PITCH_RANGE.stop)
        elif feature_name == 'duration':
            plt.xlim(0, 7)
        elif feature_name == 'onset_interval':
            plt.xlim(0, 4)

        plt.savefig(out_dir / f'datasets_{feature_name}.pdf', bbox_inches='tight')
        plt.clf()
