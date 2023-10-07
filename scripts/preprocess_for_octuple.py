#!/usr/bin/python3 python

"""Pre-process for octuple

"""

from pathlib import Path
from copy import deepcopy
from math import ceil

from miditoolkit import MidiFile
from tqdm import tqdm

from dataset import list_mmd_files_paths
from constants import OCT_MAX_BAR, MIN_NB_NOTES


if __name__ == "__main__":
    datasets = ["GiantMIDI", "POP909", "MMD"]

    for dataset in datasets:
        merged_out_dir = Path("data", f"{dataset}-short")
        if merged_out_dir.exists():
            continue
        merged_out_dir.mkdir(parents=True, exist_ok=True)
        if dataset == "MMD":
            midi_paths = list_mmd_files_paths(
                Path("data", "MMD_METADATA", "midi_audio_matches.json")
            )
        else:
            midi_paths = list(Path("data", dataset).glob("**/*.mid"))

        for i, midi_path in enumerate(
            tqdm(midi_paths, desc="PRE-PROCESSING FOR OCTUPLE")
        ):
            # Loads MIDI, merges and saves it
            midi = MidiFile(midi_path)
            ticks_per_cut = OCT_MAX_BAR * midi.ticks_per_beat * 4
            nb_cuts = ceil(midi.max_tick / ticks_per_cut)
            if nb_cuts < 2:
                continue
            midis = [deepcopy(midi) for _ in range(nb_cuts)]

            for j, track in enumerate(
                midi.instruments
            ):  # sort notes as they are not always sorted right
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
                if (
                    sum(len(track.notes) for track in midi_short.instruments)
                    < MIN_NB_NOTES
                ):
                    continue
                midi_short.dump(merged_out_dir / f"{midi_path.stem}_{j}.mid")
