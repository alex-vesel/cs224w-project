import os
import sys
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import DATA_PATH, ENCODER_NAME

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_text = pd.read_csv(os.path.join(DATA_PATH, "ogbn_products", "raw", "node-feat-text.csv.gz"), header=None).values

model = SentenceTransformer(ENCODER_NAME)
model.to(device)

embeddings = model.encode(node_text[:, 0], show_progress_bar=True)

# save embeddings as csv.gz
encoder_name_string = ENCODER_NAME.replace("/", "_")
pd.DataFrame(embeddings).to_csv(os.path.join(DATA_PATH, "ogbn_products", "raw", \
                        f"node-feat-{encoder_name_string}.csv.gz"), \
                        compression='gzip', index=False, header=False)
