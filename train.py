import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from dataloader import *
from model import *
from parameters import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
dataset = ObgnProductsWrapper(text=USE_TEXT, subset_size=SUBSET_SIZE)
dataset.to(device)

# load linear encoder which ensures all inputs to the GCN have
# the same hidden dimension
encoder = LinearEncoder(INPUT_DIM, HIDDEN_DIM)
encoder.to(device)

# load model depending on MODEL parameter
if MODEL == 'gcn':
    model = GCN(
        input_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        encoder=encoder,
        return_logits=True
    )
elif MODEL == 'gat':
    model = GAT(
        input_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM, 
        output_dim=OUTPUT_DIM, 
        num_layers=NUM_LAYERS, 
        heads=1,
        dropout=DROPOUT,
        encoder=encoder,
        emb=True
    )
elif MODEL == "graphsage":
    model = GraphSAGE(
        input_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        encoder=encoder,
        return_logits=True
    )
model.to(device)
print("Total params: ", sum(p.numel() for p in model.parameters()))

# load optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# load loss function
criterion = torch.nn.CrossEntropyLoss()

# this function performs 1 epoch of training
def train(model, dataset, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    # forward pass
    out = model(dataset.graph.x, dataset.graph.adj_t)

    # calculate loss
    out = out[dataset.split_idx['train']]
    labels = torch.squeeze(dataset.graph.y[dataset.split_idx['train']])
    loss = criterion(out, labels)
    
    # train model
    loss.backward()
    optimizer.step()

    return loss.item(), out, labels

# this function performs model evaluation on the specified split
def evaluate(model, dataset, criterion, split):
    model.eval()
    out = model(dataset.graph.x, dataset.graph.adj_t)

    # calculate loss
    out = out[dataset.split_idx[split]]
    labels = torch.squeeze(dataset.graph.y[dataset.split_idx[split]])
    loss = criterion(out, labels)

    model.train()
    return loss.item(), out, labels
    
# this function takes in model output logits and ground truth labels
# and outputs the accuracy@1
def get_accuracy(out, labels):
    out = out.argmax(dim=-1, keepdim=True)
    correct = out.eq(labels.view_as(out)).sum().item()
    out = out.cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy = correct / len(labels)
    return accuracy

# main training loop
for epoch in range(EPOCHS):
    train_loss, out, labels = train(model, dataset, optimizer, criterion, device)
    accuracy = get_accuracy(out, labels)
    print(f"Epoch: {epoch}, Train Loss:  {train_loss:.4f}, Train Accuracy: {accuracy:.4f}")
    # evaluate model every EVAL_EPOCHS
    if epoch % EVAL_EPOCHS == 0:
        val_loss, val_out, val_labels = evaluate(model, dataset, criterion, 'val')
        val_accuracy = get_accuracy(val_out, val_labels)
        print(f"Val Loss:  {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        del val_loss, val_out, val_labels
        
        
# eval on test
test_loss, test_out, test_labels = evaluate(model, dataset, criterion, 'test')
test_accuracy = get_accuracy(test_out, test_labels)
print(f"Test Loss:  {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

