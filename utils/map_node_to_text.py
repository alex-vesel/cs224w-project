import pandas as pd
import os
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import DATA_PATH

# this script aligns the node indices in the ogbn_products dataset with the text in the Amazon-3M dataset

node_asin_lookup = pd.read_csv(os.path.join(DATA_PATH, "ogbn_products/mapping/nodeidx2asin.csv.gz"))
asin_text_lookup = {}
for file_name in ["train.json", "test.json"]:
    # open json file path is DATA_PATH/Amazon-3M and add to asin_text_lookup
    with open(os.path.join(DATA_PATH, "Amazon-3M", file_name)) as f:
        for line in f:
            data = json.loads(line)
            asin_text_lookup[data['uid']] = data['title'] + " " + data['content']

# create list of text rows
node_text = []
for i, asin in enumerate(node_asin_lookup['asin']):
    node_text.append(asin_text_lookup[asin])

# create csv.gz file with text rows
node_text_df = pd.DataFrame(node_text)
node_text_df.to_csv(os.path.join(DATA_PATH, "ogbn_products_text/raw/node-feat.csv.gz"), compression='gzip', index=False, header=False)