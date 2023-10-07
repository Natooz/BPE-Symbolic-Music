#!/usr/bin/python3 python

""" Script to transform the MIDI datasets into tokens.

"""
from typing import Union, Tuple, Dict, List
from pathlib import Path
import json

from miditok.utils import merge_tracks_per_class
from miditoolkit import MidiFile
from transformers.trainer_utils import set_seed
from tqdm import tqdm
import numpy as np

from constants import (
    DATA_AUGMENTATION_OFFSETS,
    MMD_MIDI_CATS_TO_MERGE,
    MMD_VALID_PROGRAMS,
)


def is_midi_valid(
    midi: MidiFile, min_nb_tracks: int = 1, four_beats_per_bar_only: bool = False
) -> bool:
    """Returns whether a MIDI file is valid or not
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
    if four_beats_per_bar_only and any(
        ts.numerator != 4 for ts in midi.time_signature_changes
    ):
        return False

    return True


def is_midi_valid_multi(
    midi: MidiFile,
    min_nb_tracks: int = 1,
    four_beats_per_bar_only: bool = True,
    cats_to_merge: List[int] = MMD_MIDI_CATS_TO_MERGE,
    new_program_per_cat: Dict[int, int] = None,
    max_nb_of_tracks_per_inst_cat: Dict[int, int] = None,
    valid_programs: List[int] = MMD_VALID_PROGRAMS,
) -> bool:
    """Multitrack version of is_midi_valid

    :param midi: MIDI object to valid
    :param min_nb_tracks: number min of tracks (default 1 to pass everything)
    :param four_beats_per_bar_only: will discard MIDIs with time signatures other than 4/*
    :param cats_to_merge: instrument categories to merge, to give as list of indexes
            (see miditok.constants.INSTRUMENT_CLASSES), default is None for no merging
    :param new_program_per_cat: new program of the final merged tracks, to be given per
            instrument category as a dict {cat_id: program}
    :param max_nb_of_tracks_per_inst_cat: max number of tracks per instrument category,
            if the limit is exceeded for one category only the tracks with the maximum notes will be kept
    :param valid_programs: valid program numbers to keep, others will be deleted (default all)
    :return: True if the MIDI is valid, else False
    """
    merge_tracks_per_class(
        midi,
        cats_to_merge,
        new_program_per_cat,
        max_nb_of_tracks_per_inst_cat,
        valid_programs,
    )

    if not is_midi_valid(midi, min_nb_tracks, four_beats_per_bar_only):
        return False

    if len(midi.instruments) < min_nb_tracks:
        return False

    return True


def data_augmentation_dataset_pm(
    data_path: Union[Path, str],
    tokenizer=None,
    nb_octave_offset: int = None,
    nb_vel_offset: int = None,
    nb_dur_offset: int = None,
    octave_directions: Tuple[bool, bool] = (True, True),
    vel_directions: Tuple[bool, bool] = (True, True),
    dur_directions: Tuple[bool, bool] = (True, True),
    all_offset_combinations: bool = False,
    out_path: Union[Path, str] = None,
    copy_original_in_new_location: bool = True,
):
    r"""Perform data augmentation on a whole dataset, on the pitch dimension.
    Drum tracks are not augmented.
    The new created files have names in two parts, separated with a '§' character.
    Make sure your files do not have '§' in their names if you intend to reuse the information of the
    second part in some script.

    :param data_path: root path to the folder containing tokenized json files.
    :param tokenizer: tokenizer, needs to have 'Pitch' or 'NoteOn' tokens. Has to be given
            if performing augmentation on tokens (default: None).
    :param nb_octave_offset: number of pitch octaves offset to perform data augmentation.
    :param nb_vel_offset: number of velocity values
    :param nb_dur_offset: number of pitch octaves offset to perform data augmentation.
    :param octave_directions: directions to shift the pitch augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param vel_directions: directions to shift the velocity augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param dur_directions: directions to shift the duration augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param all_offset_combinations: will perform data augmentation on all the possible
            combinations of offsets. If set to False, will perform data augmentation
            only based on the original sample.
    :param out_path: output path to save the augmented files. Original (non-augmented) MIDIs will be
            saved to this location. If none is given, they will be saved in the same location an the
            data_path. (default: None)
    :param copy_original_in_new_location: if given True, the orinal (non-augmented) MIDIs will be saved
            in the out_path location too. (default: True)
    """
    if out_path is None:
        out_path = data_path
    else:
        if isinstance(out_path, str):
            out_path = Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)
    files_paths = list(Path(data_path).glob("**/*.json"))

    nb_augmentations, nb_tracks_augmented = 0, 0
    for file_path in tqdm(files_paths, desc="Performing data augmentation"):
        with open(file_path) as json_file:
            file = json.load(json_file)
            ids, programs = file["ids"], file["programs"]

        if tokenizer.unique_track:
            ids = [ids]

        # Perform data augmentation for each track
        offsets = get_offsets(
            tokenizer,
            nb_octave_offset,
            nb_vel_offset,
            nb_dur_offset,
            octave_directions,
            vel_directions,
            dur_directions,
            ids=ids,
        )
        augmented_tokens: Dict[Tuple[int, int, int], List[Union[int, List[int]]]] = {}
        for track, (_, is_drum) in zip(ids, programs):
            if is_drum:  # we dont augment drums
                continue
            aug = data_augmentation_tokens_pm(
                np.array(track),
                tokenizer,
                *offsets,
                all_offset_combinations=all_offset_combinations,
            )
            if len(aug) == 0:
                continue
            for aug_offsets, seq in aug:
                if tokenizer.unique_track:
                    augmented_tokens[aug_offsets] = seq
                    continue
                try:
                    augmented_tokens[aug_offsets].append(seq)
                except KeyError:
                    augmented_tokens[aug_offsets] = [seq]
        for i, (track, (_, is_drum)) in enumerate(
            zip(ids, programs)
        ):  # adding drums to all already augmented
            if is_drum:
                for aug_offsets in augmented_tokens:
                    augmented_tokens[aug_offsets].insert(i, track)

        # Save augmented tracks as json
        for aug_offsets, tracks_seq in augmented_tokens.items():
            pitch_tok_offset = len(tokenizer.velocities)
            vel_tok_offset = 1
            if len(tokenizer.token_ids_of_type("PitchVelDur")) > 0:
                pitch_tok_offset *= len(tokenizer.durations)
                vel_tok_offset *= len(tokenizer.durations)
            aug_offsets = (
                int(aug_offsets[0] / pitch_tok_offset),
                int(aug_offsets[1] / vel_tok_offset),
                aug_offsets[2],
            )
            if len(tracks_seq) == 0:
                continue
            suffix = "§" + "_".join(
                [
                    f"{t}{offset}"
                    for t, offset in zip(["p", "v", "d"], aug_offsets)
                    if offset != 0
                ]
            )
            saving_path = (
                file_path.parent if out_path is None else out_path
            ) / f"{file_path.stem}{suffix}.json"
            tokenizer.save_tokens(tracks_seq, saving_path, programs)
            nb_augmentations += 1
            nb_tracks_augmented += len(tracks_seq)
        if copy_original_in_new_location and out_path is not None:
            tokenizer.save_tokens(ids, out_path / f"{file_path.stem}.json", programs)

    # Saves data augmentation report, json encoded with txt extension to not mess with others json files
    with open(data_path / "data_augmentation.txt", "w") as outfile:
        json.dump(
            {
                "nb_tracks_augmented": nb_tracks_augmented,
                "nb_files_before": len(files_paths),
                "nb_files_after": len(files_paths) + nb_augmentations,
            },
            outfile,
        )


def get_offsets(
    tokenizer=None,
    nb_octave_offset: int = None,
    nb_vel_offset: int = None,
    nb_dur_offset: int = None,
    octave_directions: Tuple[bool, bool] = (True, True),
    vel_directions: Tuple[bool, bool] = (True, True),
    dur_directions: Tuple[bool, bool] = (True, True),
    ids: List[Union[int, List[int]]] = None,
) -> List[List[int]]:
    r"""Build the offsets in absolute value for data augmentation.

    :param tokenizer: tokenizer, needs to have 'Pitch' tokens.
    :param nb_octave_offset: number of pitch octaves offset to perform data augmentation.
    :param nb_vel_offset: number of velocity values
    :param nb_dur_offset: number of pitch octaves offset to perform data augmentation.
    :param octave_directions: directions to shift the pitch augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param vel_directions: directions to shift the velocity augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param dur_directions: directions to shift the duration augmentation, for up / down
            as a tuple of two booleans. (default: (True, True))
    :param ids: token ids as a list of tracks (default: None)
    :return: augmented MIDI objects.
    """
    offsets = []

    if nb_octave_offset is not None:
        # Get the maximum and lowest pitch in original track
        ids_pitch = []
        pvm_ids_vocab = tokenizer.token_ids_of_type("PitchVel")
        pvdm_ids_vocab = tokenizer.token_ids_of_type("PitchVelDur")
        pitch_ids_vocab = np.array(pvm_ids_vocab + pvdm_ids_vocab)

        for track_ids in ids:
            tt_arr = np.array(track_ids)
            ids_pitch.append(tt_arr[np.isin(tt_arr, pitch_ids_vocab)])
        max_pitch = int(
            tokenizer[int(np.max(np.concatenate(ids_pitch)))]
            .split("_")[1]
            .split("-")[0]
        )
        min_pitch = int(
            tokenizer[int(np.min(np.concatenate(ids_pitch)))]
            .split("_")[1]
            .split("-")[0]
        )

        offset_up = min(
            nb_octave_offset, (tokenizer.pitch_range.stop - 1 - max_pitch) // 12
        )
        offset_down = min(
            nb_octave_offset, (min_pitch - tokenizer.pitch_range.start) // 12
        )

        vdm_offset = len(tokenizer.velocities) * 12
        if len(pvdm_ids_vocab) > 0:
            vdm_offset *= len(tokenizer.durations)
        off = []
        if octave_directions[0]:
            off += list(range(vdm_offset, offset_up * vdm_offset + 1, vdm_offset))
        if octave_directions[1]:
            off += list(range(-offset_down * vdm_offset, 0, vdm_offset))
        offsets.append(off)

    if nb_vel_offset is not None:
        id_offset = 1
        if len(tokenizer.token_ids_of_type("PitchVelDur")) > 0:
            id_offset *= len(tokenizer.durations)
        off = []
        if vel_directions[0]:
            off += list(range(id_offset, nb_vel_offset * id_offset + 1, id_offset))
        if vel_directions[1]:
            off += list(range(-nb_vel_offset * id_offset, 0, id_offset))
        offsets.append(off)

    if nb_dur_offset is not None:
        off = []
        if dur_directions[0]:
            off += list(range(1, nb_dur_offset + 1))
        if dur_directions[1]:
            off += list(range(-nb_dur_offset, 0))
        offsets.append(off)

    return offsets


def data_augmentation_tokens_pm(
    tokens: Union[np.ndarray, List[int]],
    tokenizer,
    pitch_offsets: List[int] = None,
    velocity_offsets: List[int] = None,
    duration_offsets: List[int] = None,
    all_offset_combinations: bool = False,
) -> List[Tuple[Tuple[int, int, int], List[int]]]:
    augmented = []

    # We assume no BPE

    # Converts to np array if necessary
    if not isinstance(tokens, np.ndarray):
        tokens = np.array(tokens)

    if pitch_offsets is not None:
        # Get the maximum and lowest pitch in original track
        pitch_tokens = np.array(
            tokenizer.token_ids_of_type("PitchVel")
            + tokenizer.token_ids_of_type("PitchVelDur")
        )
        mask_pitch = np.isin(tokens, pitch_tokens)

        # Perform augmentation on pitch
        for offset in pitch_offsets:
            seq = tokens.copy()
            seq[mask_pitch] += offset
            augmented.append(((offset, 0, 0), seq))

    # Velocity augmentation
    if velocity_offsets is not None:
        vel_tokens = np.array(
            tokenizer.token_ids_of_type("PitchVel")
            + tokenizer.token_ids_of_type("PitchVelDur")
        )

        def augment_vel(
            seq_: np.ndarray, offsets_: Tuple[int, int, int]
        ) -> List[Tuple[Tuple[int, int, int], np.ndarray]]:
            mask = np.isin(seq_, vel_tokens)

            aug_ = []
            for offset_ in velocity_offsets:
                aug_seq = seq_.copy()
                aug_seq[mask] += offset_
                aug_seq[mask] = np.clip(aug_seq[mask], vel_tokens[0], vel_tokens[-1])
                aug_.append(((offsets_[0], offset_, offsets_[2]), aug_seq))
            return aug_

        if all_offset_combinations:
            for i in range(len(augmented)):
                offsets, seq_aug = augmented[i]
                augmented += augment_vel(
                    seq_aug, offsets
                )  # for already augmented midis
        augmented += augment_vel(tokens, (0, 0, 0))  # for original midi

    # Duration augmentation
    if duration_offsets is not None:
        dur_tokens = np.array(
            tokenizer.token_ids_of_type("Duration")
            + tokenizer.token_ids_of_type("PitchVelDur")
        )

        def augment_dur(
            seq_: np.ndarray, offsets_: Tuple[int, int, int]
        ) -> List[Tuple[Tuple[int, int, int], np.ndarray]]:
            mask = np.isin(seq_, dur_tokens)
            aug_ = []
            for offset_ in duration_offsets:
                aug_seq = seq_.copy()
                aug_seq[mask] += offset_
                aug_seq[mask] = np.clip(aug_seq[mask], dur_tokens[0], dur_tokens[-1])
                aug_.append(((offsets_[0], offsets_[1], offset_), aug_seq))
            return aug_

        if all_offset_combinations:
            for i in range(len(augmented)):
                offsets, seq_aug = augmented[i]
                augmented += augment_dur(
                    seq_aug, offsets
                )  # for already augmented midis
        augmented += augment_dur(tokens, (0, 0, 0))  # for original midi

    for i in range(len(augmented)):
        augmented[i] = (augmented[i][0], augmented[i][1].tolist())

    return augmented


if __name__ == "__main__":
    from miditok.data_augmentation import data_augmentation_dataset

    from dataset import list_mmd_files_paths, list_mmd_files_paths_short
    from exp_generation import experiments as exp_gen
    from exp_cla_finetune import experiments as exp_cla

    for exp in exp_gen + exp_cla:
        for baseline in exp.baselines:
            if (baseline.tokens_path / "config.txt").is_file():
                continue

            # If not already done, tokenize MIDI dataset without BPE + perform data augmentation
            if baseline.tokenization_config.bpe_vocab_size is None:
                if exp.dataset != "MMD":
                    if not baseline.tokenization.startswith("Octuple"):
                        midi_paths = list(exp.data_path_midi.glob("**/*.mid"))
                    else:
                        midi_paths = list(
                            Path(str(exp.data_path_midi) + "-short").glob("**/*.mid")
                        )
                    valid_fn = is_midi_valid
                else:
                    if not baseline.tokenization.startswith("Octuple"):
                        midi_paths = list_mmd_files_paths(
                            Path("data", "MMD_METADATA", "midi_audio_matches.json")
                        )
                    else:
                        midi_paths = list_mmd_files_paths_short(
                            Path("data", "MMD_METADATA", "midi_audio_matches.json")
                        )
                    valid_fn = is_midi_valid_multi

                baseline.tokenizer.tokenize_midi_dataset(
                    midi_paths, baseline.tokens_path, valid_fn
                )
                if (
                    baseline.tokenization[-3:] == "PVm"
                    or baseline.tokenization[-4:] == "PVDm"
                ):
                    data_augmentation_dataset_pm(
                        baseline.tokens_path,
                        baseline.tokenizer,
                        *DATA_AUGMENTATION_OFFSETS,
                    )
                elif exp.dataset != "MMD":
                    data_augmentation_dataset(
                        baseline.tokens_path,
                        baseline.tokenizer,
                        *DATA_AUGMENTATION_OFFSETS,
                    )

            # Learn and apply BPE on dataset
            else:
                set_seed(42)  # for file lim random selection
                tokens_path_no_bpe = exp.baselines[0].tokens_path
                tokens_paths = list(tokens_path_no_bpe.glob("**/*.json"))
                baseline.tokens_path.mkdir(exist_ok=True, parents=True)
                baseline.tokenizer.learn_bpe(
                    baseline.tokenization_config.bpe_vocab_size,
                    tokens_paths=tokens_paths,
                )
                baseline.tokenizer.apply_bpe_to_dataset(
                    tokens_path_no_bpe, baseline.tokens_path
                )
                baseline.tokenizer.save_params(baseline.tokens_path / "config.txt")
