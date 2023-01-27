#!/bin/bash
# Downloads and preprocesses POP909

BASEDIR=$(dirname "$0")
BASEDIR=$(dirname "$BASEDIR")
cd "$BASEDIR" || exit
mkdir -p data
cd data || exit


# Download and sort files
echo 'Downloading the POP909 dataset'
curl -LJO https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip || wget https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip
unzip POP909-Dataset-master.zip && rm POP909-Dataset-master.zip
mkdir POP909
for d in POP909-Dataset-master/POP909/*/ ; do
  mid="$(basename "$d").mid"
  mv "$d$mid" "POP909/$mid"
done
rm -r POP909-Dataset-master

# Preprocess
python scripts/preprocess_pop909.py
