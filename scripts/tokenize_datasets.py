#!/usr/bin/python3 python

""" Script to transform the MIDI datasets into tokens.

"""
from typing import Union, List, Tuple
from functools import partial
from pathlib import Path, PurePath
import json

import miditok
from miditoolkit import MidiFile
from torch import LongTensor, full, cat
from torchtoolkit.utils import seed_everything
from tqdm import tqdm

import tokenizers_
from constants import NB_SCALES_OFFSET_DATA_AUGMENTATION, PITCH_RANGE, BPE_NB_FILES_LIM


def is_midi_valid(midi: MidiFile, min_nb_tracks: int = 1, four_beats_per_bar_only: bool = True) -> bool:
    """ Returns whether a MIDI file is valid or not
    The conditions are:
        - contains the minimum number of beats given
        - contains the minimum number of tracks given
        - 4/* time signature only if four_beats_per_bar_only is True
    :param midi: MIDI object to valid
    :param min_nb_tracks: number min of tracks (default 1 to pass everything)
    :param four_beats_per_bar_only: will discard MIDIs with time signatures other than 4/*
    :return: True if the MIDI is valid, else False
    """
    if len(midi.instruments) < min_nb_tracks:
        return False
    if four_beats_per_bar_only and any(ts.numerator != 4 for ts in midi.time_signature_changes):
        return False

    return True


def data_augmentation_dataset(tokenizer: miditok.MIDITokenizer, data_path: Union[Path, PurePath, str],
                              nb_scales_offset: int, pitch_range: range):
    """Perform data augmentation on a whole dataset, on the pitch dimension.

    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param data_path: root path to the folder containing tokenized json files.
    :param nb_scales_offset: number of pitch scales to perform data augmentation.
    :param pitch_range: pitch range used to tokenize the files.
    """
    files_paths = list(Path(data_path).glob('**/*.json'))
    nb_augmentations = 0
    for file_path in tqdm(files_paths, desc='Performing data augmentation'):
        with open(file_path) as json_file:
            file = json.load(json_file)
            tokens, programs = file['tokens'][0], [file['programs'][0]]  # ONLY COVERS FIRST TRACK
        if (t_name := type(tokenizer).__name__) in ['CPWord', 'OctupleMono']:
            augmented_tokens = data_augmentation_pooling(LongTensor(tokens), tokenizer, nb_scales_offset, pitch_range,
                                                         2 if t_name == 'OctupleMono' else 0)
        elif t_name[-3:] == 'PVm' or t_name[-4:] == 'PVDm':
            augmented_tokens = data_augmentation_pm(LongTensor(tokens), tokenizer, nb_scales_offset, pitch_range)
        else:
            augmented_tokens = data_augmentation(LongTensor(tokens), tokenizer, nb_scales_offset, pitch_range)
        for offset, seq in augmented_tokens:
            tokenizer.save_tokens([seq], file_path.parent / f'{file_path.stem}_{offset}.json', programs)
            nb_augmentations += 1
    # Saves data augmentation report, json encoded with txt extension to not mess with others json files
    with open(data_path / 'data_augmentation.txt', 'w') as outfile:
        json.dump({'nb_augmentations': nb_augmentations,
                   'nb_files_before': len(files_paths),
                   'nb_files_after': len(files_paths) + nb_augmentations}, outfile)


def data_augmentation(tokens: LongTensor, tokenizer: miditok.MIDITokenizer, nb_scales_offset: int, pitch_range: range) \
        -> List[Tuple[int, List[int]]]:
    """Perform data augmentation on a sequence of tokens, on the pitch dimension.

    :param tokens: tokens to perform data augmentation on.
    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param nb_scales_offset: number of pitch scales to perform data augmentation.
    :param pitch_range: pitch range used to tokenize the files.
    :return: the several data augmentations that have been performed
    """
    pitch_tokens = LongTensor(tokenizer.vocab.tokens_of_type('Pitch'))  # within vocab, tensor to speed up operation
    tokens_pitch, tokens_pitch_idx = [], []
    for i, tok in enumerate(tokens):
        if tok in pitch_tokens:
            tokens_pitch.append(tok)
            tokens_pitch_idx.append(i)
    max_pitch = tokenizer[int(max(tokens_pitch))].split('_')[1]
    min_pitch = tokenizer[int(min(tokens_pitch))].split('_')[1]
    offset_up = min(nb_scales_offset, (pitch_range.stop - 1 - int(max_pitch)) // 12)
    offset_down = min(nb_scales_offset, (int(min_pitch) - pitch_range.start) // 12)

    augmented = []
    for i in range(offset_up):
        seq = tokens.clone()
        seq[tokens_pitch_idx] += (i + 1) * 12  # shifts pitches scale up
        augmented.append(((i + 1), seq.tolist()))
    for i in range(offset_down):
        seq = tokens.clone()
        seq[tokens_pitch_idx] -= (i + 1) * 12  # shifts pitches scale down
        augmented.append((- (i + 1), seq.tolist()))

    return augmented


def data_augmentation_pm(tokens: LongTensor, tokenizer: miditok.MIDITokenizer, nb_scales_offset: int,
                         pitch_range: range) -> List[Tuple[int, List[int]]]:
    """Perform data augmentation on a sequence of tokens, on the pitch dimension, for PVm / PVDm

    :param tokens: tokens to perform data augmentation on.
    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param nb_scales_offset: number of pitch scales to perform data augmentation.
    :param pitch_range: pitch range used to tokenize the files.
    :return: the several data augmentations that have been performed
    """
    pitch_tokens = cat([LongTensor(tokenizer.vocab.tokens_of_type('PitchVel')),
                        LongTensor(tokenizer.vocab.tokens_of_type('PitchVelDur'))])
    tokens_pitch, tokens_pitch_idx = [], []
    for i, tok in enumerate(tokens):
        if tok in pitch_tokens:
            tokens_pitch.append(tok)
            tokens_pitch_idx.append(i)
    max_pitch = tokenizer[int(max(tokens_pitch))].split('_')[1].split('-')[0]
    min_pitch = tokenizer[int(min(tokens_pitch))].split('_')[1].split('-')[0]
    offset_up = min(nb_scales_offset, (pitch_range.stop - 1 - int(max_pitch)) // 12)
    offset_down = min(nb_scales_offset, (int(min_pitch) - pitch_range.start) // 12)

    if type(tokenizer).__name__[-3:] == 'PVm':
        pm_offset = len(tokenizer.velocities)
    else:
        pm_offset = len(tokenizer.velocities) * len(tokenizer.durations)
    augmented = []
    for i in range(offset_up):
        seq = tokens.clone()
        seq[tokens_pitch_idx] += (i + 1) * 12 * pm_offset  # shifts pitches scale up
        augmented.append(((i + 1), seq.tolist()))
    for i in range(offset_down):
        seq = tokens.clone()
        seq[tokens_pitch_idx] -= (i + 1) * 12 * pm_offset  # shifts pitches scale down
        augmented.append((- (i + 1), seq.tolist()))

    return augmented


def data_augmentation_pooling(tokens: LongTensor, tokenizer: miditok.MIDITokenizer, nb_scales_offset: int,
                              pitch_range: range, pitch_voc_idx: int) -> List[Tuple[int, List[int]]]:
    """Perform data augmentation on a sequence of tokens, on the pitch dimension.
    CPWord: Pitch voc is at index 2
    Octuple: at index

    :param tokens: tokens to perform data augmentation on. (T,Z)
    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param nb_scales_offset: number of pitch scales to perform data augmentation.
    :param pitch_range: pitch range used to tokenize the files.
    :param pitch_voc_idx: idx of the pitch voc.
    :return: the several data augmentations that have been performed
    """
    special_tokens = ['PAD_None', 'SOS_None', 'EOS_None', 'MASK_None', 'Pitch_Ignore']
    special_tokens = [tokenizer.vocab[pitch_voc_idx][t] for t in special_tokens
                      if t in list(tokenizer.vocab[pitch_voc_idx].event_to_token.keys())]
    tokens_pitch, tokens_pitch_idx = [], []
    for i, tok in enumerate(tokens):
        if tok[pitch_voc_idx] not in special_tokens:
            tokens_pitch.append(tok[pitch_voc_idx])
            tokens_pitch_idx.append(i)
    max_pitch = tokenizer.vocab[pitch_voc_idx][int(max(tokens_pitch))].split('_')[1]
    min_pitch = tokenizer.vocab[pitch_voc_idx][int(min(tokens_pitch))].split('_')[1]
    offset_up = min(nb_scales_offset, (pitch_range.stop - 1 - int(max_pitch)) // 12)
    offset_down = min(nb_scales_offset, (int(min_pitch) - pitch_range.start) // 12)

    augmented = []
    for i in range(offset_up):
        seq = tokens.clone()
        seq[tokens_pitch_idx, full((len(tokens_pitch_idx), ), pitch_voc_idx)] += (i + 1) * 12  # shifts pitches scale up
        augmented.append(((i + 1), seq.tolist()))
    for i in range(offset_down):
        seq = tokens.clone()
        seq[tokens_pitch_idx, full((len(tokens_pitch_idx), ), pitch_voc_idx)] -= (i + 1) * 12  # shifts scale down
        augmented.append((- (i + 1), seq.tolist()))

    return augmented


def tokenize_dataset(tokenizer: miditok.MIDITokenizer, data_dir: Path, out_dir: Path):
    if not out_dir.exists():
        midi_paths = list(data_dir.glob('**/*.mid'))
        tokenizer.tokenize_midi_dataset(midi_paths, out_dir, partial(is_midi_valid, four_beats_per_bar_only=False))
        data_augmentation_dataset(tokenizer, out_dir, NB_SCALES_OFFSET_DATA_AUGMENTATION, PITCH_RANGE)


if __name__ == '__main__':
    from exp_gen import experiments

    '''import json  # to test PVm / PVDm data augmentation
    json_files = list(Path('data', 'POP909-merged_TSD_PVm').glob('**/*.json'))
    tokenizr = tokenizers_.TSDPVm(**experiments[0].tokenizer_params)
    for json_path in tqdm(json_files):
        with open(json_path) as f:
            tokens_ = json.load(f)
        midi = tokenizr.tokens_to_midi(tokens_['tokens'])
        midi.dump(Path('PVm_MIDIs', f'{json_path.stem}.mid'))'''

    for exp in experiments:

        previous_is_bpe = False
        previous_bpe_path = None
        for baseline in exp.baselines:
            if baseline.data_path.exists():
                continue
            # Get tokenizer
            if baseline.bpe_factor == 0:  # no BPE
                if baseline.tokenization[-3:] == 'PVm' or baseline.tokenization[-4:] == 'PVDm':
                    tokenizr = getattr(tokenizers_, baseline.tokenization)(**exp.tokenizer_params)
                else:
                    tokenizr = getattr(miditok, baseline.tokenization)(**exp.tokenizer_params)
                tokenize_dataset(tokenizr, exp.data_path_midi, baseline.data_path)
                previous_is_bpe = False
            else:  # BPE
                if not previous_is_bpe:  # happens only first time
                    tokenizr = getattr(miditok, baseline.tokenization)(**exp.tokenizer_params)
                    base_tokens_path = Path('data', f'{exp.dataset}_{baseline.tokenization}')
                    first_len = len(tokenizr)

                seed_everything(exp.seed)  # for file lim random selection
                base_tokens_path = previous_bpe_path
                tokenizr.learn_bpe(base_tokens_path, first_len * baseline.bpe_factor, baseline.data_path,
                                   files_lim=BPE_NB_FILES_LIM)
                tokenizr.apply_bpe_to_dataset(base_tokens_path, baseline.data_path)
                previous_is_bpe = True  # no need to recreate tokenizer
                previous_bpe_path = baseline.data_path
