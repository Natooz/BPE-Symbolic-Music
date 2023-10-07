"""
Constants file
"""
from miditok.constants import INSTRUMENT_CLASSES
from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available


SEED = 444

# For MMD preprocessing
MMD_VALID_PROGRAMS = list(range(-1, 48)) + list(range(56, 95))  # programs to keep in the first place (no Ensemble)
MMD_MIDI_CATS_TO_MERGE = list(range(12)) + [16]  # all but effects, and drums
MMD_PROGRAMS_TO_MERGE = [p for cat in MMD_MIDI_CATS_TO_MERGE for p in INSTRUMENT_CLASSES[cat]['program_range']]
PROGRAMS = [INSTRUMENT_CLASSES[cat]['program_range'].start for cat in range(len(INSTRUMENT_CLASSES)) if cat in
            MMD_MIDI_CATS_TO_MERGE] + [p for p in MMD_VALID_PROGRAMS if p not in MMD_PROGRAMS_TO_MERGE]
MMD_NB_GENRES = 40
MMD_NB_ARTISTS = 100

# Tokenizer params (same as MidiTok expect for new constants)
PITCH_RANGE = range(21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NB_VELOCITIES = 16
ADDITIONAL_TOKENS = {'Chord': False, 'Rest': False, 'Tempo': False, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8), 'nb_tempos': 32, 'tempo_range': (40, 250), 'time_signature_range': (8, 2),
                     "programs": PROGRAMS}
SPECIAL_TOKENS = ["PAD", "MASK", "BOS", "EOS", "SEP"]
TOKENIZER_PARAMS = {'pitch_range': PITCH_RANGE, 'beat_res': BEAT_RES, 'nb_velocities': NB_VELOCITIES,
                    'additional_tokens': ADDITIONAL_TOKENS, "special_tokens": SPECIAL_TOKENS}
TIME_DIVISION = 384
DATA_AUGMENTATION_OFFSETS = (2, 1, 0)
BPE_VOCAB_SIZES = [1000, 5000, 10000, 20000]
TOKENIZATIONS = ["TSD", "REMI"]

# Transformer config (for all models)
DIM = 512
NB_HEADS = 8
D_FFWD = DIM * 4
NB_LAYERS = 12
NB_POS_ENC_PARAMS = 2048  # params for positional encoding positions
TINY_DIM = 32  # for tiny transformer baseline

# COMMON TRAINING PARAMS
DROPOUT = 0.1
BATCH_SIZE = 64
GRAD_ACC_STEPS = 1
EVAL_ACCUMULATION_STEPS = None  # to use in case of CUDA OOM during eval
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 3.0
LABEL_SMOOTHING = 0.0
VALID_SPLIT = 0.10
TEST_SPLIT = 0.15
USE_CUDA = True
USE_MPS = False
USE_AMP = True
TORCH_COMPILE = False
TORCH_COMPILE_BACKEND = None  # default to "inductor"
TORCH_COMPILE_MODE = None
USE_GRADIENT_CHECKPOINTING = True
DDP_FIND_UNUSED_PARAMETERS = False
DDP_BUCKET_CAP_MB = None  # default to 25mb
TRAINING_STEPS = 100000
VALID_INTVL = 1000
LOG_STEPS_INTVL = 20
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 1

# TRAINING PARAMS GEN
BATCH_SIZE_GEN = BATCH_SIZE
MIN_SEQ_LEN_GEN = 256
MAX_SEQ_LEN_GEN = 384
TRAINING_STEPS_GEN = TRAINING_STEPS
LEARNING_RATE_GEN = 1e-4
LR_SCHEDULER_GEN = 'cosine_with_restarts'
WARMUP_RATIO_GEN = 0.3
VALID_SPLIT_GEN = 0.02
TEST_SPLIT_GEN = 0.10

# TRAINING PARAMS PRETRAINING
BATCH_SIZE_PT = BATCH_SIZE
MIN_SEQ_LEN_PT = 256
MAX_SEQ_LEN_PT = 384
TRAINING_STEPS_PT = TRAINING_STEPS
LEARNING_RATE_PT = 1e-4
MASK_RATIO_CLA_PT = 0.15
LR_SCHEDULER_PT = 'cosine_with_restarts'
WARMUP_RATIO_PT = 0.3

# TRAINING PARAMS CLA FT
BATCH_SIZE_CLA_FT = BATCH_SIZE
MIN_SEQ_LEN_CLA_FT = 256
MAX_SEQ_LEN_CLA_FT = 384
TRAINING_STEPS_CLA_FT = 30000
LEARNING_RATE_CLA_FT = 3e-5
LR_SCHEDULER_CLA = 'cosine_with_restarts'
WARMUP_RATIO_CLA = 0.3

# TEST PARAMS GEN
NB_INFERENCES_GEN = 512
MIN_SEQ_LEN_TEST_GEN = 256
MAX_SEQ_LEN_TEST_GEN = 384
BATCH_SIZE_TEST_GEN = 96
NUM_BEAMS = 1  # in practice the generation will use a batch size = BATCH_SIZE_TEST * NUM_BEAMS
TEMPERATURE_SAMPLING = 2.5
TOP_K = 15
TOP_P = 0.95
EPSILON_CUTOFF = None
ETA_CUTOFF = None
NB_BEATS_PROMPT_GEN = 16
NB_BEATS_CONTINUATION_MAX = 16 * 4
MIN_NB_NOTES_PROMPT_GEN = 10

# TEST PARAMS CLA
MIN_SEQ_LEN_TEST_CLA = 256
MAX_SEQ_LEN_TEST_CLA = 384
BATCH_SIZE_TEST_CLA = 96

# For CP Word and Octuple
EMBED_POOLING_SIZE = 24
MIN_NB_NOTES = 20
OCT_MAX_BAR = 30  # to shorten MIDIs


# in case no GPU is available
if not cuda_available():
    USE_AMP = USE_CUDA = False
if USE_CUDA or not mps_available():
    USE_MPS = False
