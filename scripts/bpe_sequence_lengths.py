#!/usr/bin/python3 python

"""Reads BPE tokens files to measure the variations from non-BPE

"""
from typing import List, Tuple
from pathlib import Path
import logging
from math import prod

from numpy import mean, std
from torchtoolkit.utils import seed_everything
from miditok import MIDITokenizer
from miditoolkit import MidiFile

from exp_gen import experiments, bpe_factors
from scripts.utils import reduce_list


def variation(n1, n2) -> float: return (n2 - n1) / n1 * 100


if __name__ == '__main__':
    seed_everything(777)
    (out_dir := Path('analysis', 'bpe_seq_len_variations')).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('bpe_seq_len_variations')
    logger.addHandler(logging.FileHandler(out_dir / 'bpe_seq_len_variations.log'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    file_nb_limit = 1000
    labels = ['noBPE'] + [f'bpe{f}' for f in bpe_factors[1:]] + ['PVm', 'PVDm', 'CPWord', 'Octuple']

    # Loads tokenizers
    for exp in experiments:
        for baseline in exp.baselines:
            baseline.load_tokenizer()

    def get_seq_lengths_subset(midi_path: Path, tokenizer: MIDITokenizer) -> Tuple[List[int], List[int]]:
        midi_paths = reduce_list(list(midi_path.glob('**/*.mid')), file_nb_limit)
        tok_lengths, tokens_per_beats = [], []
        for file_path in midi_paths:
            midi = MidiFile(file_path)
            tokens = tokenizer.midi_to_tokens(midi)  # We retokenize here as json files did not keep the names of mid
            if len(tokens) == 0:
                continue
            tok_lengths.append(len(tokens[0]))
            tokens_per_beats.append(tok_lengths[-1] / (midi.max_tick / midi.ticks_per_beat))
        return tok_lengths, tokens_per_beats

    all_var_lens, all_var_tpb = [[] for _ in range(len(labels))], [[] for _ in range(len(labels))]
    for ei, exp in enumerate(experiments):
        logger.debug(f'\n{exp.name}')
        original_mean_len, original_mean_tpb = 0, 0

        for bi, baseline in enumerate(exp.baselines):
            lengths, tpb = get_seq_lengths_subset(exp.data_path_midi, baseline.tokenizer)
            mean_length, mean_tpb = mean(lengths), mean(tpb)
            if bi == 0:
                original_mean_len, original_mean_tpb = mean_length, mean_tpb
            vocab_size = prod([len(v) for v in baseline.tokenizer.vocab]) \
                if isinstance(baseline.tokenizer.vocab, list) else len(baseline.tokenizer)
            var_seq_len, var_tpb = variation(original_mean_len, mean_length), variation(original_mean_tpb, mean_tpb)
            all_var_lens[bi].append(float(var_seq_len))
            all_var_tpb[bi].append(float(var_tpb))

            logger.debug(f'{baseline.name} : {mean_tpb:.2f} ± {std(tpb):.2f} tokens / beat'
                         f' - ({var_tpb:.2f}% from original tokens / beat)'
                         f' - {mean_length:.2f} ± {std(lengths):.2f} tokens / file'
                         f' - ({var_seq_len:.2f}% from original seq len)'
                         f' - vocab size: {vocab_size}')

    for type_, seq in [('var seq len', all_var_lens), ('var tokens / beat', all_var_tpb)]:
        logger.debug(f'\nMean {type_}:')
        for bi, res in enumerate(seq):
            logger.debug(f'{labels[bi]} {mean(res):.2f} ± {std(res):.2f}')
