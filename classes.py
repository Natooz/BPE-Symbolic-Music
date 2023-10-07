from pathlib import Path
from typing import List, Dict, Union
from dataclasses import dataclass
from abc import ABC

from transformers import PretrainedConfig, GPT2Config, BertConfig, TrainingArguments, Seq2SeqTrainingArguments, \
    DataCollator, GenerationConfig
import miditok

from dataset import DatasetABC
import tokenizers_


@dataclass
class DataConfig:
    valid_ratio: float
    test_ratio: float
    min_seq_len: int  # used for training
    max_seq_len: int


@dataclass
class TestingConfig:
    batch_size: int
    min_seq_len: int
    max_seq_len: int


@dataclass
class TokenizationConfig:
    tokenization: str
    tokenizer_params: Dict
    bpe_vocab_size: int = None


@dataclass
class Baseline(ABC):
    """Represents a baseline.
    Need to be overridden to have a create_dataset method.
    """
    name: str  # bpe or tokenization
    exp_name: str  # data_tokenization
    dataset: str
    seed: int
    tokenization_config: TokenizationConfig
    model_config: Union[PretrainedConfig, GPT2Config, BertConfig]
    training_config: TrainingArguments
    data_config: DataConfig
    test_config: TestingConfig
    generation_config: GenerationConfig = None
    embed_pooling_size: int = None  # as cannot be in BERTConfig / GPT2Config

    def __post_init__(self):
        tokens_path_dir_name = f"{self.dataset}_{self.tokenization}"
        if self.tokenization_config.bpe_vocab_size is not None:
            tokens_path_dir_name += f"_bpe{self.tokenization_config.bpe_vocab_size}"
        self.tokens_path = Path("data", tokens_path_dir_name)
        self.tokenizer = self.create_tokenizer()  # created with method below, called by Experiment class

        self.training_config.output_dir = str(self.run_path)  # override output dir
        self.training_config.logging_dir = self.training_config.output_dir  # for tensorboard
        self.model_config.vocab_size = len(self.tokenizer)
        self.model_config.pad = self.pad_token
        if isinstance(self.model_config, GPT2Config):
            self.model_config.bos_token_id = self.bos_token
            self.model_config.eos_token_id = self.eos_token
            self.generation_config.pad_token_id = self.pad_token
        if isinstance(self.training_config, Seq2SeqTrainingArguments):
            self.training_config.generation_config = self.generation_config

    def create_tokenizer(self) -> miditok.MIDITokenizer:
        try:
            _ = getattr(miditok, self.tokenization)
            package = miditok
        except AttributeError:
            package = tokenizers_

        try:
            tokenizer = getattr(package, self.tokenization)(params=self.tokens_path / 'config.txt')
        except FileNotFoundError:
            tokenizer = getattr(package, self.tokenization)(**self.tokenization_config.tokenizer_params)

        return tokenizer

    @property
    def tokenization(self) -> str: return self.tokenization_config.tokenization

    @property
    def run_path(self) -> Path: return Path('runs', self.exp_name, self.name)

    def __return_special_token(self, tok: str) -> int:
        if self.tokenizer.is_multi_voc:
            return self.tokenizer[0, tok]
        return self.tokenizer[tok]

    @property
    def pad_token(self) -> int:
        return self.__return_special_token('PAD_None')

    @property
    def mask_token(self) -> int:
        return self.__return_special_token('MASK_None')

    @property
    def bos_token(self) -> int:
        return self.__return_special_token('BOS_None')

    @property
    def eos_token(self) -> int:
        return self.__return_special_token('EOS_None')

    @property
    def sep_token(self) -> int:
        return self.__return_special_token('SEP_None')

    @property
    def special_tokens(self) -> List[int]:
        return [self.pad_token, self.mask_token, self.bos_token, self.eos_token, self.sep_token]

    def create_model(self):
        raise NotImplementedError

    def create_dataset(self, files_paths: List[Path], *args, **kwargs) -> DatasetABC:
        raise NotImplementedError

    def create_data_collator(self) -> DataCollator:
        raise NotImplementedError

    def __repr__(self):
        return f'{self.name} - {self.tokens_path}'


@dataclass
class Experiment:
    name: str  # dataset_tokenization
    baselines: List[Baseline]
    dataset: str

    @property
    def data_path_midi(self):
        return Path('data', self.dataset)  # original dataset path, in MIDI

    @property
    def run_path(self) -> Path: return Path('runs', self.name)

    def __str__(self): return f'{self.name} - {len(self.baselines)} baselines'

    def __repr__(self): return self.__str__()
