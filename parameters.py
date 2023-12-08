# Data parameters
DATA_PATH = 'data/'
USE_TEXT = False
SUBSET_SIZE = 100 # set to None for full dataset

# Encoder parameters
# ENCODER_NAME = "BAAI/bge-large-en-v1.5"
ENCODER_NAME = "BAAI/bge-small-en"

# Model parameters
BOW_DIM = 100
INPUT_DIM = 384
HIDDEN_DIM = 256
NUM_LAYERS = 3
OUTPUT_DIM = 47
DROPOUT = 0.2

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 10e-4