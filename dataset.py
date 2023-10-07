from pathlib import Path
from typing import Dict, Union, List, Tuple, Any
from abc import ABC
import json

from torch import LongTensor, cat, stack, full, arange
import torch
from torch.utils.data import Dataset
from miditok import MIDITokenizer
from miditoolkit import MidiFile
from tqdm import tqdm


class DatasetABC(Dataset, ABC):
    def __init__(self, samples: List[LongTensor] = None, labels: List[int] = None):
        self.samples = samples if samples is not None else []
        self.labels = labels

    def reduce_nb_samples(self, nb_samples: int):
        idx = torch.randint(0, len(self), (nb_samples,))
        self.samples = [self.samples[id_] for id_ in idx.tolist()]
        if self.labels is not None:
            self.labels = [self.labels[id_] for id_ in idx.tolist()]

    def __len__(self) -> int: return len(self.samples)

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'


class DatasetMIDI(DatasetABC):
    r"""Basic Dataset loading MIDI files.

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    :param output_labels: will output a "labels" entry in the return item. (default: False)
    """

    def __init__(
            self,
            files_paths: List[Path],
            min_seq_len: int,
            max_seq_len: int,
            tokenizer: MIDITokenizer = None,
            output_labels: bool = False,
            additional_kwargs: Dict = None,
    ):
        self.output_labels = output_labels
        self.additional_kwargs = additional_kwargs
        samples = []

        for file_path in tqdm(files_paths, desc=f"Loading data: {files_paths[0].parent}"):
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)["ids"]
            if not tokenizer.unique_track:
                tokens = tokens[0]  # first track
            if len(tokens) < min_seq_len:
                continue  # tokens of this file not long enough

            # Cut tokens in samples of appropriate length
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                i += len(samples[-1])  # could be replaced with max_seq_len

        super().__init__(samples)

    def __getitem__(self, idx) -> Dict[str, LongTensor]:
        item = {"input_ids": self.samples[idx]}
        if self.output_labels:
            item["labels"] = self.samples[idx]
        if self.additional_kwargs is not None:
            for key, val in self.additional_kwargs:
                item[key] = val
        return item


class DatasetClaComposer(DatasetABC):
    r"""Dataset for composer classification.
    Only for GiantMIDI
    NSP: https://github.com/huggingface/transformers/blob/main/src/transformers/data/datasets/language_modeling.py
    For NSP, would need to add a SEP token to tokenizer vocab, and either mix sequences within Dataset
    (not dynamic) or override DataCollatorForLanguageModeling to shuffle sequences (dynamic).

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(
            self,
            files_paths: List[Path],
            min_seq_len: int,
            max_seq_len: int,
            max_nb_composers: int,
            tokenizer: MIDITokenizer = None
    ):
        labels = []
        composers = {}  # stores composer_name: [samples_idx]

        for file_path in tqdm(files_paths, desc=f'Preparing data: {files_paths[0].parent}'):
            # Check file is good
            parts = file_path.name.split(', ')
            if len(parts) < 4:
                continue

            # Load tokens
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids'][0]  # first track
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
                sample = LongTensor(tokens[i:i + max_seq_len])
                composers[composer].append(sample)
                i += max_seq_len

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
            labels += len(composers[composer]) * [i]
            self.nb_composers += 1

        del composers
        super().__init__(samples, labels)

    def __getitem__(self, idx) -> Dict[str, Union[LongTensor, int]]:
        return {"input_ids": self.samples[idx],
                "labels": self.labels[idx]}


class DatasetClaEmotion(DatasetABC):
    r"""Dataset for emotion classification, with the EMOPIA dataset.

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(
            self,
            files_paths: List[Path],
            min_seq_len: int,
            max_seq_len: int,
            tokenizer: MIDITokenizer = None
    ):
        samples = []
        samples_labels = []

        for file_path in tqdm(files_paths, desc=f'Preparing data: {files_paths[0].parent}'):
            # Load tokens
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids'][0]  # first track
            if len(tokens) < min_seq_len:
                continue  # tokens of this file not long enough

            # Get emotion label
            label = int(file_path.name[1]) - 1  # Q1/Q2/Q3/Q4

            # Cut tokens in samples of appropriate length
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                samples_labels.append(label)
                i += max_seq_len

        super().__init__(samples, samples_labels)

    def __getitem__(self, idx) -> Dict[str, Union[LongTensor, int]]:
        return {"input_ids": self.samples[idx], "labels": self.labels[idx]}


class DatasetClaMMD(DatasetABC):
    r"""Dataset for genre of artist classification with MMD

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(
            self,
            files_paths: List[Path],
            min_seq_len: int,
            max_seq_len: int,
            max_nb_labels: int,
            genres: bool = False,
            artists: bool = False,
            tokenizer: MIDITokenizer = None,
    ):
        labels_idx = []
        labels = {}  # stores label_name: [samples (token_ids)]
        if genres:
            file_path = Path("data", "MMD_METADATA", "MMD_scraped_genre.jsonl")
            key = "genre"
        elif artists:
            file_path = Path("data", "MMD_METADATA", "MMD_scraped_title_artist.jsonl")
            key = "title_artist"
        else:
            raise ValueError
        md5_to_labels = {}
        with open(file_path) as file:
            for row in file:
                entry = json.loads(row)
                if len(entry[key][0]) == 0:
                    continue
                if genres:
                    md5_to_labels[entry["md5"]] = entry[key][0][0]
                else:
                    md5_to_labels[entry["md5"]] = entry[key][0][1]

        for file_path in tqdm(files_paths, desc=f'Preparing data: {files_paths[0].parent}'):
            md5 = file_path.stem.split("_")[0]  # split for MMD-short
            if not md5 in md5_to_labels:
                continue
            label = md5_to_labels[md5]
            if label not in labels:
                labels[label] = []

            # Load tokens
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids']
            if len(tokens) < min_seq_len:
                continue  # tokens of this file not long enough

            # Cut tokens in samples of appropriate length
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                sample = LongTensor(tokens[i:i + max_seq_len])
                labels[label].append(sample)
                i += max_seq_len

        # Keep only max_nb_labels, ones with most samples
        samples = []
        labels_sorted_per_nb_samples = sorted(labels.keys(), key=lambda x: len(labels[x]))
        for i, label in enumerate(labels_sorted_per_nb_samples[-max_nb_labels:]):
            samples += labels[label]
            labels_idx += len(labels[label]) * [i]

        del labels
        super().__init__(samples, labels_idx)

    def __getitem__(self, idx) -> Dict[str, Union[LongTensor, int]]:
        return {"input_ids": self.samples[idx],
                "labels": self.labels[idx]}


class DataCollatorStandard:
    def __init__(
            self,
            pad_token: int,
            bos_token: int = None,
            eos_token: int = None,
            pad_on_left: bool = False,
            shift_labels: bool = False,
            labels_pad_idx: int = -100,
            add_bos_eos_to_labels: bool = False,
            inputs_kwarg_name: str = "input_ids",
            labels_kwarg_name: str = "labels",
    ):
        """Multifunction data collator, that can pad the sequences (right or left), add BOS and EOS tokens.
        Input_ids will be padded with the pad token given, while labels will be padded with -100.

        :param pad_token: PAD token
        :param bos_token: BOS token (default: None).
        :param eos_token: EOS token (default: None).
        :param pad_on_left: will pad sequence on the left (default: False).
        :param shift_labels: will shift inputs and labels for autoregressive training / teacher forcing.
        :param labels_pad_idx: padding idx for labels (default: -100).
        :param add_bos_eos_to_labels: will add BOS and/or EOS tokens to the labels (default: False).
        :param inputs_kwarg_name: name of dict / kwarg key for inputs (default: "input_ids").
        :param inputs_kwarg_name: name of dict / kwarg key for inputs (default: "labels_").
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_on_left = pad_on_left
        self.shift_labels = shift_labels
        self.labels_pad_idx = labels_pad_idx
        self.add_bos_eos_to_labels = add_bos_eos_to_labels
        self.inputs_kwarg_name = inputs_kwarg_name
        self.labels_kwarg_name = labels_kwarg_name

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, LongTensor]:
        out_batch = {}
        pad_on_left = batch[0]["pad_on_left"] if "pad_on_left" in batch[0] else self.pad_on_left

        # Add BOS and EOS tokens + PAD to inputs
        x = None
        if self.inputs_kwarg_name in batch[0]:
            _add_bos_eos_tokens_to_batch(batch, self.inputs_kwarg_name, bos_tok=self.bos_token, eos_tok=self.eos_token)
            x = _pad_batch(batch, self.pad_token, self.inputs_kwarg_name, pad_on_left)

        # Add BOS and EOS tokens + PAD labels
        y = None
        if self.labels_kwarg_name in batch[0]:
            # generation or language modeling
            if isinstance(batch[0][self.labels_kwarg_name], LongTensor):
                _add_bos_eos_tokens_to_batch(batch, self.labels_kwarg_name,
                                             bos_tok=self.bos_token, eos_tok=self.eos_token)
                y = _pad_batch(batch, self.labels_pad_idx, self.labels_kwarg_name, pad_on_left)
            # classification
            else:
                y = LongTensor([item[self.labels_kwarg_name] for item in batch])

        # Shift labels
        if self.shift_labels:  # otherwise it's handled in models such as GPT2LMHead
            if x is not None:
                inputs = x
            elif y is not None:
                inputs = y
            else:
                raise ValueError("Either inputs or labels have to be specified by the Dataset.")
            x = inputs[:-1]
            y = inputs[1:]

        # Add inputs / labels to output batch
        if x is not None:
            out_batch[self.inputs_kwarg_name] = x
        if y is not None:
            out_batch[self.labels_kwarg_name] = y

        # Create attention mask (just for padding, causality is handled in models)
        attention_mask = (x != self.pad_token).int()
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[..., 0]  # (N,T,Z) --> (N,T)
        out_batch["attention_mask"] = attention_mask

        return out_batch


class DataCollatorClaPreTrain:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all the same length.
    Inspired from transformers.DataCollatorForLanguageModeling
    """

    def __init__(
            self,
            pad_token: int,
            bos_token: int,
            eos_token: int,
            mask_token: int,
            sep_token: int,
            vocab_size: Union[int, List[int]],
            special_tokens: List[int],
            mlm_probability: float = 0.15,
            nsp_probability: float = 0.5,
            sentence_b_ratio: float = 0.5,
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.sep_token = LongTensor([sep_token])
        self.vocab_size = vocab_size
        self.special_tokens = LongTensor(special_tokens)
        self.mlm_probability = mlm_probability
        self.nsp_probability = nsp_probability
        self.sentence_b_ratio = sentence_b_ratio

    def __call__(self, batch: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        _add_bos_eos_tokens_to_batch(batch, bos_tok=self.bos_token, eos_tok=self.eos_token)
        # Determine idx to mix for NSP
        batch, token_type_ids, next_sentence_label = _nsp_swap(batch,
                                                               self.nsp_probability,
                                                               self.sentence_b_ratio,
                                                               self.sep_token,
                                                               self.pad_token)

        # Pad and mask them
        masked_inputs, original_input = self.torch_mask_tokens(_pad_batch(batch, self.pad_token))
        attention_mask = (masked_inputs != self.pad_token).int()
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[..., 0]  # (N,T,Z) --> (N,T)

        # If special token mask has been preprocessed, pop it from the dict.
        batch = {"input_ids": masked_inputs,
                 "labels": original_input,
                 "token_type_ids": token_type_ids,
                 "next_sentence_label": next_sentence_label,
                 "attention_mask": attention_mask}
        return batch

    def torch_mask_tokens(self, inputs: LongTensor) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.isin(inputs, self.special_tokens)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        if inputs.dim() == 3:
            random_words = []
            for i in range(inputs.shape[-1]):
                random_words.append(torch.randint(self.vocab_size[i], inputs.shape[:-1]).long())
            random_words = stack(random_words, -1)
        else:
            random_words = torch.randint(self.vocab_size, labels.shape).long()
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorContrastive:
    def __init__(self, pad_token: int, bos_token: int, eos_token: int):
        """Collator for contrastive learning.
        The labels are ranks (arange()).

        :param pad_token: pas token
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, LongTensor]:
        _add_bos_eos_tokens_to_batch(batch, bos_tok=self.bos_token, eos_tok=self.eos_token)
        x = _pad_batch(batch, self.pad_token)  # .repeat(2, 1)  # (N*2,T)
        attention_mask = (x != self.pad_token).int()
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[..., 0]  # (N,T,Z) --> (N,T)
        return {"input_ids": x, "labels": arange(x.size(0)).long(), "attention_mask": attention_mask}  # rank


def _add_bos_eos_tokens_to_batch(
        batch: List[Dict[str, LongTensor]],
        dict_key: str = "input_ids",
        bos_tok: int = None,
        eos_tok: int = None
):
    if bos_tok is None and eos_tok is None:
        return

    (sos_shape := list(batch[0][dict_key].shape))[0] = 1  # (1) or (1,Z)
    for i in range(len(batch)):
        if bos_tok is not None and eos_tok is not None:
            batch[i][dict_key] = cat([full(sos_shape, bos_tok),
                                      batch[i][dict_key],
                                      full(sos_shape, eos_tok)], dim=0).long()
        elif bos_tok is not None:
            batch[i][dict_key] = cat([full(sos_shape, bos_tok), batch[i][dict_key]], dim=0).long()
        else:  # EOS not None
            batch[i][dict_key] = cat([batch[i][dict_key], full(sos_shape, eos_tok)], dim=0).long()


def _pad_batch(
        batch: List[Dict[str, LongTensor]],
        pad_token: int,
        dict_key: str = "input_ids",
        pad_on_left: bool = False
) -> LongTensor:
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    length_of_first = batch[0][dict_key].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x[dict_key].size(0) == length_of_first for x in batch)
    if are_tensors_same_length:
        return stack([e[dict_key] for e in batch], dim=0).long()

    # Creating the full tensor and filling it with our data.
    if pad_on_left:
        return _pad_left([e[dict_key] for e in batch], pad_token)
    else:
        return torch.nn.utils.rnn.pad_sequence(
            [e[dict_key] for e in batch],
            batch_first=True,
            padding_value=pad_token
        ).long()


def _pad_left(batch: List[LongTensor], pad_token: int) -> LongTensor:
    # Here the sequences are padded to the left, so that the last token along the time dimension
    # is always the last token of each seq, allowing to efficiently generate by batch
    batch = [torch.flip(seq, dims=(0,)) for seq in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_token)  # (N,T)
    batch = torch.flip(batch, dims=(1,)).long()
    return batch


def _nsp_swap(
        examples: List[Dict[str, LongTensor]],
        nsp_ratio: float,
        seq_b_ratio: float,
        sep_token: Union[int, LongTensor],
        pad_token: Union[int, LongTensor],
) -> Tuple[List[Dict[str, LongTensor]], LongTensor, LongTensor]:
    # Determine idx to mix for NSP
    nb_mixed = int(len(examples) * nsp_ratio)
    token_type_ids = [full((len(examples[idx]["input_ids"]) + 1,), 0).long() for idx in range(len(examples))]
    next_sentence_label = full((len(examples),), 0).long()
    new_next_idx = arange(len(examples))
    if nb_mixed > 1:
        # create permutations / pairs of swapped seq_a - seq_b
        permutations = torch.randperm(nb_mixed)
        while any(equal := permutations == arange(nb_mixed)):
            idx_equal = torch.where(equal)[0]  # list of idx to mix together
            if len(idx_equal) == 1:
                idx_to_swap = torch.multinomial(cat([permutations[:idx_equal[0]],
                                                     permutations[idx_equal[0] + 1:]]).float(), 1).long()
                permutations[idx_equal[0]] = idx_to_swap
                permutations[idx_to_swap] = idx_equal[0]
            else:
                permutations[idx_equal] = permutations[idx_equal[torch.randperm(len(idx_equal))]]  # only betw those eq
        samples_to_mix_idx = torch.multinomial(arange(len(examples)).float(), nb_mixed).long()
        new_next_idx[samples_to_mix_idx] = samples_to_mix_idx.clone()[permutations]

    # Swap them and prepare labels / token_type_ids
    # SEP token need to be added at the end (before padding) as we may otherwise swap sequences of
    # different lengths and add additional SEP tokens to some
    examples_copy = [e["input_ids"].clone() for e in examples]
    (sep_shape := list(examples[0]["input_ids"].shape))[0] = 1  # (1) or (1,Z)
    sep_token_ = full(sep_shape, int(sep_token))
    for idx, idx_next in enumerate(new_next_idx):
        sep_idx = int(len(examples[idx]["input_ids"]) * seq_b_ratio)
        len_seq_b = len(examples[idx]["input_ids"]) - sep_idx
        len_next_seq = len(examples[idx_next]["input_ids"]) - 1  # -1 because of BOS token
        if len_seq_b > len_next_seq:
            sep_idx = len(examples[idx]["input_ids"]) - len_next_seq
            len_seq_b = len_next_seq
        token_type_ids[idx] = cat([token_type_ids[idx][:sep_idx + 1], full((len_seq_b,), 1).long()]).long()
        if idx != idx_next:  # meaning seq_b is not seq_a's second part
            next_sentence_label[idx] = 1

        examples[idx]["input_ids"] = cat([examples_copy[idx][:sep_idx],
                                          sep_token_,
                                          examples_copy[idx_next][-len_seq_b:]]).long()
        examples[idx]["labels"] = cat([examples_copy[idx][:sep_idx],
                                       sep_token_,
                                       examples_copy[idx_next][-len_seq_b:]]).long()

    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, True, float(pad_token)).long()  # stack and pad
    return examples, token_type_ids, next_sentence_label


def list_mmd_files_paths(match_file_path: Path = Path('results', 'midi_audio_matches.json')) -> List[Path]:
    with open(match_file_path) as file:
        matches = json.load(file)
    return [Path('data', 'MMD', md5[0], md5[1], md5[2], md5).with_suffix('.mid') for md5 in matches.keys()]


def list_mmd_files_paths_short(match_file_path: Path = Path('results', 'midi_audio_matches.json')) -> List[Path]:
    with open(match_file_path) as file:
        matches = json.load(file)
    all_paths = Path("data", "MMD-short").glob("**/*.mid")
    return [path for path in all_paths if path.stem.split("_")[0] in matches]
