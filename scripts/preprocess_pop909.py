#!/usr/bin/python3 python

"""Pre-process the POP909 dataset: merges tracks
Make sure to rename the original POP909 directory to 'POP909-original'

"""

from pathlib import Path
from copy import deepcopy

from miditok.utils import merge_tracks
from miditoolkit import MidiFile
from tqdm import tqdm


if __name__ == "__main__":
    # Make sure to rename the original POP909 directory to 'POP909-original'
    midi_paths = list(Path("data", "POP909-original").glob("**/*.mid"))
    merged_out_dir = Path("data", "POP909")
    merged_out_dir.mkdir(parents=True, exist_ok=True)

    for i, midi_path in enumerate(
        tqdm(midi_paths, desc="PRE-PROCESSING POP909 DATASET")
    ):
        # Loads MIDI, merges and saves it
        midi = MidiFile(midi_path)
        for track in midi.instruments:  # sort notes as they are not always sorted right
            track.notes.sort(key=lambda x: x.start)
        merge_tracks(merged_midi := deepcopy(midi))
        merged_midi.dump(merged_out_dir / midi_path.name)
