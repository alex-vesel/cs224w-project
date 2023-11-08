# Data parameters
DATA_PATH = 'data/'
USE_TEXT = True

# Encoder parameters
USE_LM_ENCODER = True
ENCODER_NAME = "BAAI/bge-large-en-v1.5"

# Model parameters
INPUT_DIM = 768
HIDDEN_DIM = 256
NUM_LAYERS = 3
OUTPUT_DIM = 47
DROPOUT = 0.2

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 10e-4