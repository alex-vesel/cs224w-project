import torch

from dataloader import *
from model import *
from parameters import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
dataset = ObgnProductsWrapper(split='train', text=USE_TEXT)
dataset.to(device)

# load model
model = GCN(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
)

# load optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# load loss function
criterion = torch.nn.NLLLoss()

def train(model, dataset, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    # forward pass
    out = model(dataset.graph.x, dataset.graph.adj_t, from_text=USE_TEXT)

    # calculate loss
    loss = criterion(out[dataset.split_idx['train']], dataset.graph.y[dataset.split_idx['train']])
    loss.backward()
    optimizer.step()

    return loss.item()


# train
for epoch in range(EPOCHS):
    train_loss = train(model, dataset, optimizer, criterion, device)
    print(f"Epoch: {epoch}, Train Loss:  {train_loss:.4f}")

