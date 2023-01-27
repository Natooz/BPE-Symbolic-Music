"""
Constants file
"""
SEED = 444

# Tokenizer params (same as MidiTok expect for new constants)
PITCH_RANGE = range(21, 109)
BEAT_RES = {(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}
NB_VELOCITIES = 8
ADDITIONAL_TOKENS = {'Chord': False, 'Rest': False, 'Tempo': False, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8), 'nb_tempos': 32, 'tempo_range': (40, 250), 'time_signature_range': (8, 2)}
TOKENIZER_PARAMS = {'pitch_range': PITCH_RANGE, 'beat_res': BEAT_RES, 'nb_velocities': NB_VELOCITIES,
                    'additional_tokens': ADDITIONAL_TOKENS, 'sos_eos': True}
TIME_DIVISION = 384
NB_SCALES_OFFSET_DATA_AUGMENTATION = 2
BPE_NB_FILES_LIM = 1500

# For classification
MAX_NB_COMPOSERS = 10

# Transformer config (for generator)
DIM = 512
NB_HEADS = 8
D_FFWD = 2048
NB_LAYERS = 10
NB_POS_ENC_PARAMS = 2048  # params for positional encoding positions

# Transformer config (for classifier)
CLA_DIM = 768
CLA_NB_HEADS = 12
CLA_D_FFWD = 2048
CLA_NB_LAYERS = 10
CLA_NB_POS_ENC_PARAMS = 2048  # params for positional encoding positions
CLA_LARGE_DIM = 1024
CLA_LARGE_NB_HEADS = 16
CLA_LARGE_D_FFWD = 3072
CLA_LARGE_NB_LAYERS = 18
CLA_LARGE_NB_POS_ENC_PARAMS = 2048  # params for positional encoding positions


# Training params
DROPOUT = 0.1
BATCH_SIZE = 16
GRAD_ACC_STEPS = 1
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 3.0
LABEL_SMOOTHING = 0.0
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.3
VALID_SPLIT = 0.35
TEST_SPLIT = 0.15  # unused
USE_CUDA = True
USE_AMP = True
TRAINING_STEPS = 100000
EARLY_STOP_STEPS = 15000  # nb of steps to stop training if no increase of valid loss
ITERATOR_KWARGS = {'early_stop_steps': EARLY_STOP_STEPS}
VALID_INTVL = 30
NB_VALID_STEPS = 5
LOG_INTVL = 10
MIN_SEQ_LEN = 384
MAX_SEQ_LEN = 460

# GEN TEST PARAMS
NB_INFERENCES_TEST = 1024
MAX_SEQ_LEN_TEST = 1024
BATCH_SIZE_TEST = 32
NUM_BEAMS = 1  # in practice the generation will use a batch size = BATCH_SIZE_TEST * NUM_BEAMS
TOP_P = 0.9

# TRAINING PARAMS DIS / CLA
CLA_PRE_TRAINING_STEPS = 60000
CLA_TRAINING_STEPS = 100000
CLA_BATCH_SIZE = 24
CLA_LARGE_BATCH_SIZE = 24
CLA_PT_LEARNING_RATE = 1e-6
CLA_FT_LEARNING_RATE = 5e-7
CLA_EARLY_STOP = 25000
RANDOM_RATIO_RANGE = (0.01, 0.15)

# For CP Word and Octuple
EMBED_SIZES_CP = [32, 64, 512, 128, 128]  # fam, pos / bar, pitch, vel, dur
EMBED_SIZES_OCTUPLE = [512] * 5  # pitch, vel, dur, pos, bar
OCT_MAX_BAR = 30  # to shorten MIDIs

# For metrics
CONSISTENCY_WINDOWS_LEN = 16  # in beats
