# A Semantically Sweet Syndicate — Sentence Embedding Models and GNNs

This repository contains modules for running experiments using language models as encoders for various GNNs. The directory structure is represented as follows:

- **/:** the top-level directory contains train.py, which is the main entry point and training script. The paremeters used by all the code is defined in parameters.py, also in this directory.
- **/dataloader/:** this directory contains the dataloader obgn_products_wrapper.py, which wraps the PyG obgn-products dataset to enable merging of the precomputed text embeddings. 
- **/model/:** this directory contains 3 GNN models: GCN, GAT, and GraphSAGE. More details on these can be found in our Medium article.
- **/utils/:** this directory contains utility scripts. encode_text.py reads the raw natural language product node descriptions and encodes them using a HuggingFace language model. map_node_to_text.py maps the node idx for the PyG array to an actual product ID to align the text descriptions.

To run the training procedure, download the processed data from here and place the unzipped folder in a directory called /data/. Then simply run the training script using:
```
python ./train.py
```
You can modify the training and model hyperparameters in the parameters.py file. The training run will print the train loss and accuracy every epoch, as well as periodically evaluating on the validation set. At the end of the training run, the model will be evaluated on the test set.
