# LibriSpeech dataset Config
TRAIN_PREFIX = "train"
TRAIN_SPLITS = ["train.clean.100"]
DEV_PREFIX = "dev"
DEV_SPLITS = ["validation.clean"]
TEST_PREFIX = "test"
TEST_SPLITS = ["test.clean"]


# Audio Config
TARGET_SAMPLING_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80


# Tokenizer Config
VOCAB_SIZE = 1000
SPM_MODEL_PATH = "spm.model"


# Model Config
MODEL_DIM = 256
STATE_DIM = 16
CONV_KERNEL = 4
NUM_LAYERS = 12


# Training Config
BATCH_SIZE = 16
NUM_EPOCHS = 10
MODEL_PATH = "model.pt"
LEARNING_RATE = 3e-4

# Inference Config
MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0
