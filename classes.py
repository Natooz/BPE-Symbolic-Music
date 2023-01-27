from pathlib import Path
from typing import List, Union

from transformers import GPT2Config, BertConfig
import miditok

from model import GenTransformer, GenTransformerPooling, ClassifierTransformer, ClassifierTransformerPooling
from constants import *
import tokenizers_


class ModelConfig:
    def __init__(self, dim: int = DIM,
                 nb_heads: int = NB_HEADS,
                 d_ffwd: int = D_FFWD,
                 nb_layers: int = NB_LAYERS,
                 nb_pos_enc_params: int = NB_POS_ENC_PARAMS,
                 embed_sizes: List[int] = None):
        self.dim = dim
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.d_ffwd = d_ffwd
        self.nb_pos_enc_params = nb_pos_enc_params

        self.embed_sizes = embed_sizes  # for CP Word and Octuple


class TrainingConfig:
    def __init__(self, use_cuda: bool = USE_CUDA,
                 use_amp: bool = USE_AMP,
                 batch_size: int = BATCH_SIZE,
                 grad_acc_steps: int = GRAD_ACC_STEPS,
                 learning_rate: float = LEARNING_RATE,
                 weight_decay: float = WEIGHT_DECAY,
                 gradient_clip_norm: float = GRADIENT_CLIP_NORM,
                 label_smoothing: float = LABEL_SMOOTHING,
                 dropout: float = DROPOUT,
                 valid_split: float = VALID_SPLIT,
                 test_split: float = TEST_SPLIT,
                 training_steps: int = TRAINING_STEPS,
                 warmup_ratio: float = WARMUP_RATIO,
                 iterator_kwargs: dict = ITERATOR_KWARGS,
                 valid_intvl: int = VALID_INTVL,
                 nb_valid_steps: int = NB_VALID_STEPS,
                 log_intvl: int = LOG_INTVL,
                 min_seq_len: int = MIN_SEQ_LEN,
                 max_seq_len: int = MAX_SEQ_LEN):
        self.use_cuda = use_cuda
        self.use_amp = use_amp
        self.batch_size = batch_size
        self.grad_acc_steps = grad_acc_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clip_norm = gradient_clip_norm
        self.label_smoothing = label_smoothing
        self.dropout = dropout
        self.valid_split = valid_split
        self.test_split = test_split
        self.training_steps = training_steps
        self.warmup_ratio = warmup_ratio
        self.iterator_kwargs = iterator_kwargs
        self.valid_intvl = valid_intvl
        self.nb_valid_steps = nb_valid_steps
        self.log_intvl = log_intvl
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len


class TestingConfig:
    def __init__(self, batch_size: int = BATCH_SIZE_TEST,
                 max_seq_len: int = MAX_SEQ_LEN_TEST,
                 nb_inferences_test: int = NB_INFERENCES_TEST,
                 num_beams: int = NUM_BEAMS,
                 top_p: float = TOP_P):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.nb_inferences_test = nb_inferences_test
        self.num_beams = num_beams
        self.top_p = top_p


class Baseline:
    """Represents a baseline
    """
    def __init__(self, name: str, exp_name: str, dataset: str, seed: int, tokenization: str, bpe_factor: int,
                 model_config: ModelConfig):
        self.name = name  # bpe or tokenization
        self.exp_name = exp_name  # data_tokenization (remi/tsd), for run path as exp.tokenization can be different
        self.dataset = dataset  # can be different from the one in exp_name, for example with short
        self.seed = seed  # is used when splitting generated data in train / test sets

        self.bpe_factor = bpe_factor
        self.tokenization = tokenization
        self.tokenizer = None
        self.model_config = model_config

    def load_tokenizer(self):
        if self.tokenization[-3:] == 'PVm' or self.tokenization[-4:] == 'PVDm':
            self.tokenizer = getattr(tokenizers_, self.tokenization)(params=self.data_path / 'config.txt')
        else:
            self.tokenizer = getattr(miditok, self.tokenization)(params=self.data_path / 'config.txt')

    @property
    def data_path(self) -> Path: return Path('data', f'{self.exp_name}' + (f'_{self.name}' if self.name != '' else ''))

    @property
    def run_path(self) -> Path: return Path('runs', self.exp_name, self.name)

    @property
    def run_path_classifier(self) -> Path: return Path('runs_classifier', self.exp_name, self.name)

    @property
    def gen_data_path(self) -> Path: return self.run_path / 'gen'

    @property
    def is_embed_pooling(self) -> bool: return isinstance(self.tokenizer.vocab, list)

    @property
    def pad_token(self) -> int:
        return self.tokenizer.vocab[0]['PAD_None'] if self.is_embed_pooling else self.tokenizer['PAD_None']

    @property
    def sos_token(self) -> int:
        return self.tokenizer.vocab[0]['SOS_None'] if self.is_embed_pooling else self.tokenizer['SOS_None']

    @property
    def eos_token(self) -> int:
        return self.tokenizer.vocab[0]['EOS_None'] if self.is_embed_pooling else self.tokenizer['EOS_None']

    def __repr__(self):
        return f'{self.name} - {self.data_path}'


class Experiment:
    def __init__(self, baselines: List[Baseline], dataset: str, tokenization: str, seed: int,
                 cla_model_conf: ModelConfig, gen_train_conf: TrainingConfig, cla_pre_train_conf: TrainingConfig,
                 cla_train_conf: TrainingConfig, gen_test_conf: TestingConfig,
                 tokenizer_params: dict = TOKENIZER_PARAMS):
        self.name = f'{dataset}_{tokenization}'  # dataset_tokenization
        self.run_path = Path('runs', self.name)
        self.data_path_midi = Path('data', dataset)  # original dataset path, in MIDI
        self.baselines = baselines
        self.dataset = dataset
        self.tokenizer_params = tokenizer_params  # used when tokenizing datasets only
        self.seed = seed

        self.cla_model_conf = cla_model_conf
        self.gen_train_conf = gen_train_conf
        self.cla_pre_train_conf = cla_pre_train_conf
        self.cla_train_conf = cla_train_conf
        self.gen_test_conf = gen_test_conf

    def create_gen(self, baseline: Baseline) -> Union[GenTransformer, GenTransformerPooling]:
        """Creates the generative model for the experiment.
        The model must implement the `forward_train` and `infer` methods.
        """
        config_d = GPT2Config(vocab_size=len(baseline.tokenizer.vocab),
                              n_positions=baseline.model_config.nb_pos_enc_params,
                              n_embd=baseline.model_config.dim, n_layer=baseline.model_config.nb_layers,
                              n_head=baseline.model_config.nb_heads, n_inner=baseline.model_config.d_ffwd,
                              resid_pdrop=self.gen_train_conf.dropout, embd_pdrop=self.gen_train_conf.dropout,
                              attn_pdrop=self.gen_train_conf.dropout, pad_token_id=baseline.pad_token,
                              bos_token_id=baseline.sos_token, eos_token_id=baseline.eos_token)
        if baseline.is_embed_pooling:
            num_classes = [len(v) for v in baseline.tokenizer.vocab]
            return GenTransformerPooling(config_d, num_classes, baseline.model_config.embed_sizes)
        else:
            return GenTransformer(config_d)

    def create_classifier(self, baseline: Baseline, num_labels: int = None, pre_train: bool = False):
        """Creates the model for the experiment.
        The model must implement the `forward_train` and `infer` methods.
        """
        model_conf = self.cla_model_conf
        train_conf = self.cla_train_conf
        config = BertConfig(vocab_size=len(baseline.tokenizer), hidden_size=model_conf.dim,
                            num_hidden_layers=model_conf.nb_layers, num_attention_heads=model_conf.nb_heads,
                            intermediate_size=model_conf.d_ffwd, hidden_dropout_prob=train_conf.dropout,
                            attention_probs_dropout_prob=train_conf.dropout,
                            max_position_embeddings=model_conf.nb_pos_enc_params,
                            type_vocab_size=2, pad_token_id=baseline.pad_token, num_labels=num_labels)
        if baseline.is_embed_pooling:
            num_classes = [len(v) for v in baseline.tokenizer.vocab]
            return ClassifierTransformerPooling(config, num_classes, model_conf.embed_sizes, pre_train)
        return ClassifierTransformer(config, pre_train)

    def __str__(self): return f'{self.name} - {len(self.baselines)} baselines'

    def __repr__(self): return self.__str__()
