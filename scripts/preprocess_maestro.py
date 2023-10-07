#!/usr/bin/python3 python

"""Remove the tree dir of the Maestro dataset and put all mid files in one dir.
"""

import shutil
from pathlib import Path

from tqdm import tqdm


if __name__ == "__main__":
    # Make sure to rename the original POP909 directory to 'POP909-original'
    final_path = Path("data", "Maestro")
    temp_path = Path("data", "Maestro_tmp")
    shutil.move(final_path, temp_path)
    final_path.mkdir(parents=True, exist_ok=True)
    midi_paths = list(temp_path.glob("**/*.mid"))

    for midi_path in tqdm(midi_paths, desc="MOVING MAESTRO DATASET"):
        midi_path.rename(final_path / midi_path.name)

    shutil.rmtree(temp_path)
