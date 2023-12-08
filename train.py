import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from dataloader import *
from model import *
from parameters import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
dataset = ObgnProductsWrapper(text=USE_TEXT, subset_size=SUBSET_SIZE)
dataset.to(device)

# load LM encoder
if USE_TEXT:
    encoder = LMEncoder(ENCODER_NAME)
else:
    encoder = LinearEncoder(BOW_DIM, INPUT_DIM)
encoder.to(device)

# load model
model = GCN(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    encoder=encoder,
    return_logits=True
)
model.to(device)

# load optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# load loss function
criterion = torch.nn.CrossEntropyLoss()

def train(model, dataset, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    # forward pass
    out = model(dataset.graph.x, dataset.graph.adj_t)

    # calculate loss
    out = out[dataset.split_idx['train']]
    labels = torch.squeeze(dataset.graph.y[dataset.split_idx['train']])
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    return loss.item(), out, labels

def evaluate(out, labels):
    out = out.argmax(dim=-1, keepdim=True)
    correct = out.eq(labels.view_as(out)).sum().item()
    out = out.cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = correct / len(labels)
    return accuracy

# train
for epoch in range(EPOCHS):
    train_loss, out, labels = train(model, dataset, optimizer, criterion, device)
    accuracy = evaluate(out, labels)
    print(f"Epoch: {epoch}, Train Loss:  {train_loss:.4f}, Train Accuracy: {accuracy:.4f}")

