#!/usr/bin/python3 python

"""Pre-process for octuple

"""

from pathlib import Path
from copy import deepcopy
from math import ceil

from miditoolkit import MidiFile
from tqdm import tqdm

from exp_gen import datasets
from constants import OCT_MAX_BAR


if __name__ == '__main__':
    for dataset in datasets:
        midi_paths = list(Path('data', dataset).glob('**/*.mid'))
        (merged_out_dir := Path('data', f'{dataset}-short')).mkdir(parents=True, exist_ok=True)

        for i, midi_path in enumerate(tqdm(midi_paths, desc='PRE-PROCESSING FOR OCTUPLE')):
            # Loads MIDI, merges and saves it
            midi = MidiFile(midi_path)
            ticks_per_cut = OCT_MAX_BAR * midi.ticks_per_beat * 4
            nb_cuts = ceil(midi.max_tick / ticks_per_cut)
            if nb_cuts < 2:
                continue
            midis = [deepcopy(midi) for _ in range(nb_cuts)]

            for j, track in enumerate(midi.instruments):  # sort notes as they are not always sorted right
                track.notes.sort(key=lambda x: x.start)
                for midi_short in midis:  # clears notes from shorten MIDIs
                    midi_short.instruments[j].notes = []
                for note in track.notes:
                    cut_id = note.start // ticks_per_cut
                    note_copy = deepcopy(note)
                    note_copy.start -= cut_id * ticks_per_cut
                    note_copy.end -= cut_id * ticks_per_cut
                    midis[cut_id].instruments[j].notes.append(note_copy)

            # Saving MIDIs
            for j, midi_short in enumerate(midis):
                midi_short.dump(merged_out_dir / f'{midi_path.stem}_{j}.mid')
