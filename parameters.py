# Data parameters
DATA_PATH = 'data/'
USE_TEXT = False
SUBSET_SIZE = 800000 # set to None for full dataset

# Encoder parameters
# ENCODER_NAME = "BAAI/bge-base-en-v1.5"
ENCODER_NAME = "BAAI/bge-small-en"

# Model parameters
MODEL = 'graphsage'
if USE_TEXT:
    if ENCODER_NAME == "BAAI/bge-base-en-v1.5":
        INPUT_DIM = 768
    elif ENCODER_NAME == "BAAI/bge-small-en":
        INPUT_DIM = 384
else:
    INPUT_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 3
OUTPUT_DIM = 47
DROPOUT = 0.

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 10e-4
TRAIN_SIZE = 0.7
VAL_SIZE = 0.1
EVAL_EPOCHS = 10