#!/bin/bash
# Downloads the datasets

BASEDIR=$(dirname "$0")
BASEDIR=$(dirname "$BASEDIR")
cd "$BASEDIR" || exit
mkdir -p data
cd data || exit


# POP909
echo 'Downloading the POP909 dataset'
curl -LJO https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip || wget https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip
unzip POP909-Dataset-master.zip && rm POP909-Dataset-master.zip
mkdir POP909-original
for d in POP909-Dataset-master/POP909/*/ ; do
  mid="$(basename "$d").mid"
  mv "$d$mid" "POP909-original/$mid"
done
rm -r POP909-Dataset-master
python ../scripts/preprocess_pop909.py


# EMOPIA
echo 'Downloading the EMOPIA dataset'
curl -LJO https://zenodo.org/record/5090631/files/EMOPIA_1.0.zip?download=1 || wget https://zenodo.org/record/5090631/files/EMOPIA_1.0.zip?download=1
unzip EMOPIA_1.0.zip && rm EMOPIA_1.0.zip
mkdir EMOPIA
for mid in EMOPIA_1.0/midis/* ; do
  mi="$(basename "$mid")"
  mv "$mid" "EMOPIA/$mi"
done
rm -r EMOPIA_1.0

# MAESTRO (v3.0.0) (only MIDI)
curl -LJO https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip || wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip && rm maestro-v3.0.0-midi.zip
mv maestro-v3.0.0 Maestro
# shellcheck disable=SC2044
for file in $(find "Maestro" -type f -name "*.midi"); do
    mv -- "$file" "${file%.midi}.mid"
done
python ../scripts/preprocess_maestro.py

# MMD
python ../scripts/clean_mmd.py

# PREPROCESS FOR OCTUPLE
python ../scripts/preprocess_for_octuple.py