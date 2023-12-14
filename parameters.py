# Data parameters
DATA_PATH = 'data/'     # path to ogbn_products dataset parent folder
USE_TEXT = True         # flag of whether to use text embeddings or BoW representation
SUBSET_SIZE = 800000    # number of nodes to be used, set to None for full dataset

# Encoder parameters
ENCODER_NAME = "BAAI/bge-base-en-v1.5"  # encoder name

# Model parameters
MODEL = 'gcn'           # set model choice {gcn, gat, graphsage}
if USE_TEXT:
    if ENCODER_NAME == "BAAI/bge-base-en-v1.5":
        INPUT_DIM = 768
    elif ENCODER_NAME == "BAAI/bge-small-en":
        INPUT_DIM = 384
else:
    INPUT_DIM = 100
HIDDEN_DIM = 256        # hidden dimension of network
NUM_LAYERS = 3          # number of message passing layers
OUTPUT_DIM = 47         # output dimension (47 for ogbn-products)
DROPOUT = 0.            # dropout percent

# Training parameters
EPOCHS = 100            # number of training epochs
LEARNING_RATE = 10e-4   # learning rate
TRAIN_SIZE = 0.7        # proportion of dataset that is training
VAL_SIZE = 0.1          # proportion of dataset that is validation
EVAL_EPOCHS = 10        # proportion of dataset that is test