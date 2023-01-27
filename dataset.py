from pathlib import Path
from typing import Tuple, List
from abc import ABC
import json

from torch import LongTensor, stack, cat, full, flip
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtoolkit.utils import randomize_tensor
from miditok import MIDITokenizer
from miditoolkit import MidiFile
from tqdm import tqdm


class MyDataset(Dataset, ABC):
    def __init__(self, samples=None):
        self.samples = samples if samples is not None else []

    def __len__(self) -> int: return len(self.samples)

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'


class MIDIDatasetGen(MyDataset):
    r"""Dataset for generator training

    :param data_path: path containing the real data to load, ex: 'data/death_metal_dataset'.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param padding_token: padding token, usually 0.
    :param tokenizer: tokenizer object, to use when fake_data_path is a list of MIDI paths. (default: None)
    """

    def __init__(self, data_path: Path, min_seq_len: int, max_seq_len: int, padding_token: int, sos_token: int,
                 tokenizer: MIDITokenizer = None):
        self.pad_token = padding_token
        self.sos_token = sos_token
        min_seq_len -= 1  # as SOS token will be added
        max_seq_len -= 1
        samples = []
        as_midi = False
        files_paths = list(Path(data_path).glob(f'**/*.json'))
        if len(files_paths) == 0:
            files_paths = list(Path(data_path).glob(f'**/*.mid'))
            as_midi = True

        for file_path in tqdm(files_paths, desc=f'Preparing data {data_path.name}'):
            if as_midi:
                tokens = tokenizer.midi_to_tokens(MidiFile(file_path))[0]  # first track, (T) or (T,Z)
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['tokens'][0]  # first track
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                i += len(samples[-1])  # could be replaced with max_seq_len

        super().__init__(samples)

    def collate_fn(self, batch: List[LongTensor]) -> Tuple[LongTensor, LongTensor]:
        batch = pad_sequence(batch, batch_first=True, padding_value=self.pad_token)  # (N,T) or (N,T,Z)
        (sos_shape := list(batch.shape))[1] = 1  # (N,1) or (N,1,Z)
        batch = cat([full(sos_shape, self.sos_token), batch], dim=1)  # adds sos token to every samples
        return batch[:, :-1], batch[:, 1:]

    def collate_fn_infer(self, batch: List[LongTensor]) -> LongTensor:
        # Here the sequences are padded to the left, so that the last element along the time dimension
        # is always the last of each seq, allowing to efficiently generate by batch
        sos_shape = (1,) if batch[0].dim() == 1 else (1, batch[0].shape[-1])  # (1) or (1,Z)
        batch = [flip(cat([full(sos_shape, self.sos_token), seq], dim=0), dims=(0, )) for seq in batch]
        batch = pad_sequence(batch, batch_first=True, padding_value=self.pad_token)  # (N,T) or (N,T,Z)
        batch = flip(batch, dims=(1, )).long()
        return batch

    def __getitem__(self, idx) -> LongTensor:
        return self.samples[idx]


class MIDIDatasetClassification(MyDataset):
    r"""Dataset for composer classification.
    Only made for GiantMIDI

    :param data_path: path containing the real data to load, ex: 'data/death_metal_dataset'.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param padding_token: padding token, usually 0.
    :param sos_tok: Start Of Sequence (SOS) token, that will be inserted at the beginning of each sample
    :param eos_tok: End Of Sequence (EOS) token, that will be inserted at the end of each sample
    :param tokenizer: tokenizer object, to use when fake_data_path is a list of MIDI paths. (default: None)
    :param random_range: token range of random tokens. (default: None)
    :param random_ratio: ratio of randomized tokens (within the masking ratio), has to be comprised within [O,1]
    :param random_ratio_range: if random_ratio is None, a random ratio will randomly be chosen between. You
                                can set a range for this ratio, between 0 and 1 (default: None)
    """

    def __init__(self, data_path: Path, min_seq_len: int, max_seq_len: int, max_nb_composers: int, padding_token: int,
                 sos_tok: int, eos_tok: int, tokenizer: MIDITokenizer = None, **kwargs):
        self.pad_token = padding_token
        self.sos_tok = sos_tok
        self.eos_tok = eos_tok
        self.samples_composer_idx = []
        min_seq_len -= 2  # considering SOS and EOS tokens
        max_seq_len -= 2
        composers = {}  # stores composer_name: [samples_idx]
        loading_midi = tokenizer is not None
        files_paths = list(Path(data_path).glob(f'**/*.{"mid" if loading_midi else "json"}'))

        self.random_range = kwargs.get('random_range', None)
        self.random_ratio = kwargs.get('random_ratio', None)
        self.random_ratio_range = kwargs.get('random_ratio_range', None)

        count = 0  # current nb of samples
        for file_path in tqdm(files_paths, desc=f'Preparing data {data_path.name}'):
            # Check file is good
            parts = file_path.name.split(', ')
            if len(parts) < 4:
                continue

            # Load tokens
            if loading_midi:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['tokens'][0]  # first track
            if len(tokens) < min_seq_len:
                continue  # tokens of this file not long enough

            # Get composer name and store it if not already done
            composer = f'{parts[0]} {parts[1]}'
            if composer not in composers:
                composers[composer] = []

            # Cut tokens in samples of appropriate length
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                composers[composer].append(sample := LongTensor(tokens[i:i + max_seq_len]))
                i += len(sample)  # could be replaced with max_seq_len
                count += 1

        '''# Remove composers < min_nb_samples
        composers_list = list(self.composers.keys())  # all of them, as a list
        for composer in composers_list:
            if len(self.composers[composer]) < min_nb_samples_per_composer:
                del self.composers[composer]'''

        # Keep only max_nb_composers, ones with most samples
        samples = []
        composers_sorted_per_nb_samples = sorted(composers.keys(), key=lambda x: len(composers[x]))
        self.nb_composers = 0
        for i, composer in enumerate(composers_sorted_per_nb_samples[-max_nb_composers:]):
            samples += composers[composer]
            self.samples_composer_idx += len(composers[composer]) * [i]
            self.nb_composers += 1

        del composers
        super().__init__(samples)

    def collate_fn(self, batch: List[Tuple[LongTensor, int]]) -> Tuple[LongTensor, LongTensor]:
        (sos_shape := list(batch[0][0].shape))[0] = 1  # (1) or (1,Z)
        x = [cat([full(sos_shape, self.sos_tok), seq, full(sos_shape, self.eos_tok)], dim=0) for seq, _ in batch]
        x = pad_sequence(x, batch_first=True, padding_value=self.pad_token).long()  # (N,T)
        y = LongTensor([item[1] for item in batch])
        return x, y  # (N,T) and (N)

    def collate_fn_infer(self, batch: List[Tuple[LongTensor, int]]) -> Tuple[LongTensor, LongTensor]:
        return self.collate_fn(batch)

    def collate_pre_train(self, batch: List[Tuple[LongTensor, int]]) -> Tuple[LongTensor, LongTensor]:
        """Randomize some random token positions, to pre-train a non-causal LM.

        :param batch: a batch of token sequences, as a list of N tuples.
        :return:
        """
        (sos_shape := list(batch[0][0].shape))[0] = 1  # (1) or (1,Z)
        if len(sos_shape) == 1:
            x = [cat([full(sos_shape, self.sos_tok),
                      randomize_tensor(seq, self.random_range, self.random_ratio, self.random_ratio_range),
                      full(sos_shape, self.eos_tok)], dim=0) for seq, _ in batch]
        else:
            randomized = []
            for seq, _ in batch:  # over seq
                randomized.append(stack([randomize_tensor(seq[:, t], self.random_range[t], self.random_ratio,
                                                          self.random_ratio_range) for t in range(sos_shape[-1])], -1))
            x = [cat([full(sos_shape, self.sos_tok), seq, full(sos_shape, self.eos_tok)], dim=0) for seq in randomized]
        x = pad_sequence(x, batch_first=True, padding_value=self.pad_token).long()  # (N,T)
        y = [cat([full(sos_shape, self.sos_tok), seq, full(sos_shape, self.eos_tok)], dim=0) for seq, _ in batch]
        y = pad_sequence(y, batch_first=True, padding_value=self.pad_token).long()
        return x, y  # (N,T) and (N,T)

    def __getitem__(self, idx) -> Tuple[LongTensor, int]:
        return self.samples[idx], self.samples_composer_idx[idx]
