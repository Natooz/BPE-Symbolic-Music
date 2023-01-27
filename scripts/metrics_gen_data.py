#!/usr/bin/python3 python

"""Runs metrics

"""
from pathlib import Path
from typing import Union
import json
import logging

from miditok import TSD
from miditoolkit import MidiFile
from tqdm import tqdm
from torchtoolkit.utils import seed_everything
import numpy as np

from exp_gen import experiments, datasets
from tokenize_datasets import is_midi_valid
from constants import TOKENIZER_PARAMS, CONSISTENCY_WINDOWS_LEN
from metrics import notes_density, overlapping_area_hist, overlapping_area_kde, TokenComprehension, consistency,\
    note_density_evolution
from scripts.utils import reduce_list, mean_std


def note_duration_quantized(note_, dur_bins_ticks) -> int:
    duration_ticks = note_.end - note_.start
    return dur_bins_ticks[np.argmin(np.abs(dur_bins_ticks - duration_ticks))]


def read_midi(path: Path) -> Union[MidiFile, None]:
    try:
        midi = MidiFile(path)
    except:
        return None
    if not is_midi_valid(midi, four_beats_per_bar_only=False):
        return None
    tokenizer.preprocess_midi(midi)
    return midi


if __name__ == '__main__':
    seed_everything(777)
    (out_dir := Path('analysis', 'metrics_gen_data')).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('metrics_gen_data')
    logger.addHandler(logging.FileHandler(out_dir / 'metrics_gen_data.log'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    # Loads tokenizers
    for exp in experiments:
        for baseline in exp.baselines:
            baseline.load_tokenizer()

    tokenizer = TSD(**TOKENIZER_PARAMS)
    max_beat_res = max(tokenizer.beat_res.values())
    file_nb_limit = 15000

    pitches = {dataset: [] for dataset in datasets}
    durations = {dataset: [] for dataset in datasets}
    velocities = {dataset: [] for dataset in datasets}
    onsets = {dataset: [] for dataset in datasets}
    onset_intervals = {dataset: [] for dataset in datasets}  # continuous
    note_densities = {dataset: [] for dataset in datasets}  # continuous
    note_densities_evol = {dataset: [] for dataset in datasets}
    pitch_sim_data = {dataset: [] for dataset in datasets}
    vel_sim_data = {dataset: [] for dataset in datasets}
    dur_sim_data = {dataset: [] for dataset in datasets}

    # Get distributions of datasets
    if (dist_path := out_dir / 'datasets_features_dists.json').exists():
        with open(dist_path) as file:
            dists = json.load(file)
        pitches, durations, velocities = dists['pitches'], dists['durations'], dists['velocities']
        onsets = dists['onsets']
        onset_intervals = dists['onset_intervals']
        note_densities = dists['note_densities']
        note_densities_evol = dists['note_densities_evol']
        pitch_sim_data = dists['pitch_sim_data']
        vel_sim_data = dists['vel_sim_data']
        dur_sim_data = dists['dur_sim_data']
    else:
        for di, dataset in enumerate(datasets):
            files_paths = list(Path('data', dataset).glob('**/*.mid'))
            for file_path in tqdm(files_paths, desc=f'Parsing features of {dataset} dataset'):
                midi_ = read_midi(file_path)
                if midi_ is None:
                    continue
                if midi_.ticks_per_beat not in tokenizer.durations_ticks:
                    tokenizer.durations_ticks[midi_.ticks_per_beat] = \
                        np.array([(beat * rs + pos) * midi_.ticks_per_beat // rs
                                  for beat, pos, rs in tokenizer.durations])

                note_densities[dataset].append(notes_density(midi_.instruments[0].notes, midi_.ticks_per_beat))
                note_densities_evol[dataset] += note_density_evolution(midi_.instruments[0].notes,
                                                                       midi_.ticks_per_beat)
                pitch_sim, vel_sim, dur_sim = consistency(midi_.instruments[0].notes, midi_.ticks_per_beat,
                                                          CONSISTENCY_WINDOWS_LEN)
                pitch_sim_data[dataset] += pitch_sim
                vel_sim_data[dataset] += vel_sim
                dur_sim_data[dataset] += dur_sim
                for n, note in enumerate(midi_.instruments[0].notes):
                    pitches[dataset].append(note.pitch)
                    velocities[dataset].append(int(note.velocity))
                    duration_quantized = note_duration_quantized(note, tokenizer.durations_ticks[midi_.ticks_per_beat])
                    durations[dataset].append(int(duration_quantized / midi_.ticks_per_beat * max_beat_res))  # nb pos
                    onsets[dataset].append(int((note.start % midi_.ticks_per_beat) /
                                               midi_.ticks_per_beat * max_beat_res))
                    if n >= len(midi_.instruments[0].notes) - 1:
                        continue  # is the last note
                    onset_inter = (midi_.instruments[0].notes[n + 1].start - note.start) / midi_.ticks_per_beat
                    if 0 < onset_inter <= 4:  # notes played together or too far apart are discarded
                        onset_intervals[dataset].append(onset_inter)
        # Saves the distributions
        with open(out_dir / 'datasets_features_dists.json', 'w') as outfile:
            json.dump({'pitches': pitches, 'durations': durations, 'velocities': velocities,
                       'onsets': onsets, 'onset_intervals': onset_intervals, 'note_densities': note_densities,
                       'note_densities_evol': note_densities_evol, 'pitch_sim_data': pitch_sim_data,
                       'vel_sim_data': vel_sim_data, 'dur_sim_data': dur_sim_data},
                      outfile)

    for dataset in datasets:
        logger.debug(f'\n{dataset}')
        logger.debug(f'Note density evolution data-16beats: {mean_std(np.abs(np.array(note_densities_evol[dataset])))}')
        logger.debug(f'OA data-16beats pitch: {mean_std(pitch_sim_data[dataset])}')
        logger.debug(f'OA data-16beats velocity: {mean_std(vel_sim_data[dataset])}')
        logger.debug(f'OA data-16beats duration: {mean_std(dur_sim_data[dataset])}')

    # Compare with gen data
    for exp in experiments:
        logger.debug(f'\n{exp.name}')
        for baseline in exp.baselines:
            pitches_b, velocities_b, durations_b, onsets_b, onset_intervals_b, note_densities_b = [], [], [], [], [], []
            oas_pitch_pg, oas_vel_pg, oas_dur_pg, dif_note_density_pg = [], [], [], []
            tc = TokenComprehension(baseline.tokenizer)

            # Load dist if already exist, else compute them
            if (dist_path := out_dir / f'{exp.name}_{baseline.name}_dists.json').exists():
                with open(dist_path) as file:
                    dists = json.load(file)
                pitches_b, durations_b, velocities_b = dists['pitches'], dists['durations'], dists['velocities']
                onsets_b = dists['onsets']
                onset_intervals_b = dists['onset_intervals']
                note_densities_b = dists['note_densities']
                oas_pitch_pg = dists['oas_pitch_pg']
                oas_vel_pg = dists['oas_vel_pg']
                oas_dur_pg = dists['oas_dur_pg']
                dif_note_density_pg = dists['dif_note_density_pg']
                tc.load(out_dir / f'{exp.name}_{baseline.name}_tc.csv')

            else:
                files_paths = list(baseline.gen_data_path.glob('**/*.mid'))
                files_paths = reduce_list(files_paths, file_nb_limit)
                for file_path in tqdm(files_paths, desc=f'Parsing features of {exp.name} - {baseline.name} gen data'):
                    midi_ = read_midi(file_path)
                    if midi_ is None:
                        continue
                    if midi_.ticks_per_beat not in tokenizer.durations_ticks:
                        tokenizer.durations_ticks[midi_.ticks_per_beat] = \
                            np.array([(beat * rs + pos) * midi_.ticks_per_beat // rs
                                      for beat, pos, rs in tokenizer.durations])

                    note_densities_b.append(notes_density(midi_.instruments[0].notes, midi_.ticks_per_beat))
                    for n, note in enumerate(midi_.instruments[0].notes):
                        pitches_b.append(note.pitch)
                        velocities_b.append(int(note.velocity))
                        duration_quantized = note_duration_quantized(note,
                                                                     tokenizer.durations_ticks[midi_.ticks_per_beat])
                        durations_b.append(int(duration_quantized / midi_.ticks_per_beat * max_beat_res))  # tik -> beat
                        onsets_b.append(int((note.start % midi_.ticks_per_beat) / midi_.ticks_per_beat * max_beat_res))
                        if n >= len(midi_.instruments[0].notes) - 1:
                            continue  # is the last note
                        onset_inter = (midi_.instruments[0].notes[n + 1].start - note.start) / midi_.ticks_per_beat
                        if 0 < onset_inter <= 4:  # notes played together or too far apart are discarded
                            onset_intervals_b.append(onset_inter)

                    # Consistency between prompt and first beats generated
                    prompt_pitches, prompt_velocities, prompt_durations = [], [], []
                    if len(midi_.instruments[1].notes) == 0:  # empty prompt
                        continue
                    for note in midi_.instruments[1].notes:
                        prompt_pitches.append(note.pitch)
                        prompt_velocities.append(int(note.velocity))
                        duration_quantized = note_duration_quantized(note,
                                                                     tokenizer.durations_ticks[midi_.ticks_per_beat])
                        prompt_durations.append(int(duration_quantized / midi_.ticks_per_beat * max_beat_res))
                    continuation_pitches, continuation_velocities, continuation_durations = [], [], []
                    for note in midi_.instruments[0].notes:
                        if note.start / midi_.ticks_per_beat >= CONSISTENCY_WINDOWS_LEN:
                            break
                        continuation_pitches.append(note.pitch)
                        continuation_velocities.append(note.velocity)
                        duration_quantized = note_duration_quantized(note,
                                                                     tokenizer.durations_ticks[midi_.ticks_per_beat])
                        continuation_durations.append(int(duration_quantized / midi_.ticks_per_beat * max_beat_res))
                    if len(continuation_pitches) == 0:  # continuation with no notes within first
                        oas_pitch_pg.append(0)
                        oas_vel_pg.append(0)
                        oas_dur_pg.append(0)
                    else:
                        oas_pitch_pg.append(overlapping_area_hist(np.array(prompt_pitches),
                                                                  np.array(continuation_pitches)))
                        oas_vel_pg.append(overlapping_area_hist(np.array(prompt_velocities),
                                                                np.array(continuation_velocities)))
                        oas_dur_pg.append(overlapping_area_hist(np.array(prompt_durations),
                                                                np.array(continuation_durations)))
                    dif_note_density_pg.append(abs(notes_density(midi_.instruments[0].notes, midi_.ticks_per_beat) -
                                                   notes_density(midi_.instruments[1].notes, midi_.ticks_per_beat)))

                    # Token comprehension
                    with open(file_path.with_suffix('.json')) as file:
                        tokens = json.load(file)['tokens'][0]
                        if baseline.bpe_factor > 0:
                            tokens = baseline.tokenizer.decompose_bpe(tokens)
                    tc(tokens)

                # Saving distributions
                with open(out_dir / f'{exp.name}_{baseline.name}_dists.json', 'w') as outfile:
                    json.dump({'pitches': pitches_b, 'durations': durations_b, 'velocities': velocities_b,
                               'onsets': onsets_b, 'onset_intervals': onset_intervals_b,
                               'note_densities': note_densities_b,
                               'oas_pitch_pg': oas_pitch_pg, 'oas_vel_pg': oas_vel_pg, 'oas_dur_pg': oas_dur_pg,
                               'dif_note_density_pg': dif_note_density_pg}, outfile)
                tc.save(out_dir / f'{exp.name}_{baseline.name}_tc.csv', reset_after=False)

            # compute OA + log it
            logger.debug(f'\n{exp.name} - {baseline.name}')
            logger.debug(f'OA dataset-gen pitch (hist intersection): '
                         f'{overlapping_area_hist(np.array(pitches[exp.dataset]), np.array(pitches_b)):.2f}')
            logger.debug(f'OA dataset-gen velocity (hist intersection): '
                         f'{overlapping_area_hist(np.array(velocities[exp.dataset]), np.array(velocities_b)):.2f}')
            logger.debug(f'OA dataset-gen duration (hist intersection): '
                         f'{overlapping_area_hist(np.array(durations[exp.dataset]), np.array(durations_b)):.2f}')
            '''logger.debug(f'OA dataset-gen onsets (hist intersection): '
                         f'{overlapping_area_hist(np.array(onsets[exp.dataset]), np.array(onsets_b)):.2f}')'''
            oa = overlapping_area_kde(np.array(onset_intervals[exp.dataset]), np.array(onset_intervals_b))
            logger.debug(f'OA dataset-gen onset intervals (kde): {oa:.2f}')
            oa = overlapping_area_kde(np.array(note_densities[exp.dataset]), np.array(note_densities_b))
            logger.debug(f'OA dataset-gen note densities (kde): {oa:.2f}')

            # Log consistencies
            logger.debug(f'OA prompt-continuation pitch: {mean_std(oas_pitch_pg)}')
            logger.debug(f'OA prompt-continuation velocity: {mean_std(oas_vel_pg)}')
            logger.debug(f'OA prompt-continuation duration: {mean_std(oas_dur_pg)}')
            logger.debug(f'DIF prompt-continuation note density: {mean_std(dif_note_density_pg)}')
            tc.analyze(logger)
