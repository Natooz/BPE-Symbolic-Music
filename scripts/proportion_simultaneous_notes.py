#!/usr/bin/python3 python

"""Pre-process the POP909 dataset: merges tracks
Make sure to rename the original POP909 directory to 'POP909-original'

"""

from typing import List
from pathlib import Path

from miditoolkit import MidiFile
from miditok import TSD
from tqdm import tqdm

from constants import (
    TOKENIZER_PARAMS,
    PITCH_RANGE,
    NB_VELOCITIES,
    ADDITIONAL_TOKENS,
    SPECIAL_TOKENS,
)

BEAT_RES_16 = {(0, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS_16 = {
    "pitch_range": PITCH_RANGE,
    "beat_res": BEAT_RES_16,
    "nb_velocities": NB_VELOCITIES,
    "additional_tokens": ADDITIONAL_TOKENS,
    "special_tokens": SPECIAL_TOKENS,
}


def measure_proportion(midi: MidiFile) -> List[float]:
    proportions = []
    for track in midi.instruments:  # sort notes as they are not always sorted right
        track.notes.sort(key=lambda x: (x.start, x.end, x.velocity))

        nb_simultaneous = 0
        current_notes = 1
        current_vel_end = (-1, -1, -1)
        for note in track.notes:
            if (note.start, note.end, note.velocity) != current_vel_end:
                if current_notes > 1:
                    nb_simultaneous += current_notes
                    current_notes = 1
                current_vel_end = (note.start, note.end, note.velocity)

            else:
                current_notes += 1

        proportions.append(nb_simultaneous / len(track.notes))
    return proportions


if __name__ == "__main__":
    tokenizer_32 = TSD(**TOKENIZER_PARAMS)
    tokenizer_16 = TSD(**TOKENIZER_PARAMS_16)
    datasets = ["POP909", "Maestro", "GiantMIDI", "MMD", "EMOPIA"]
    files_lim = 15000

    for dataset in datasets:
        midi_paths = list(Path("data", dataset).glob("**/*.mid")) + list(
            Path("data", dataset).glob("**/*.midi")
        )
        if len(midi_paths) > files_lim:
            midi_paths = midi_paths[:files_lim]

        ppt_tick = []
        ppt_prepro_32 = []
        ppt_prepro_16 = []
        for i, midi_path in enumerate(
            tqdm(midi_paths, desc=f"Analyzing simultaneous notes ({dataset})")
        ):
            try:
                midi_ = MidiFile(midi_path)
            except:
                continue

            if midi_.ticks_per_beat < 8:
                continue
            ppt_tick += measure_proportion(midi_)

            tokenizer_32.preprocess_midi(midi_)
            ppt_prepro_32 += measure_proportion(midi_)

            tokenizer_16.preprocess_midi(midi_)
            ppt_prepro_16 += measure_proportion(midi_)

        print(
            f"{dataset} - proportion of simult. notes (ticks): {sum(ppt_tick) / len(ppt_tick)}"
        )
        print(
            f"{dataset} - proportion of simult. notes (preprocess 32): {sum(ppt_prepro_32) / len(ppt_prepro_32)}"
        )
        print(
            f"{dataset} - proportion of simult. notes (preprocess 16): {sum(ppt_prepro_16) / len(ppt_prepro_16)}"
        )
