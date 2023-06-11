import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transformations

# Create a Bidirectional Lstm Network 
class BLSTM(nn.Module):
    def __init__(self, imp_emb_dim, hidden_units, n_layers, output_classes):
        super(BLSTM, self).__init__()

        self.n_layers = n_layers
        self.hidden_units = hidden_units

        self.bilstm = nn.LSTM(input_size=imp_emb_dim,
                               hidden_size=hidden_units,
                               num_layers=n_layers,
                               batch_first=True,
                               bidirectional=True)
        
        self.fc = nn.Linear(hidden_units*imp_emb_dim*2, output_classes)

    def forward(self, x):

        # Initialize the hidden state and cell state first for bidrectional lstm
        h0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_units).to(device)
        c0 = torch.zeros(self.n_layers*2, x.size(0), self.hidden_units).to(device)
        
        # Forward Propagation
        out, _ = self.bilstm(x, (h0, c0))
        out_flatten = torch.flatten(out, 1,-1)
        x = self.fc(out_flatten)

        return x


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# HyperParameters
n_class = 10
n_layer = 10
hidden_size = 5
emb_dim = 28
batch_size = 64
max_seq_length = 12
learning_rate = 0.001
num_epochs = 5
load_model = True

# Load Data
# download and load the data from pytorch sample datasets
# https://pytorch.org/vision/0.8/datasets.html

#train dataloader
train_dataset = datasets.MNIST(root='dataset', train=True, transform=transformations.ToTensor(), download=True)
train_datloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# test dataloader
test_dataset = datasets.MNIST(root='dataset', train=False, transform=transformations.ToTensor(), download=True)
test_datloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Initialise Network
model = BLSTM(emb_dim, hidden_size, n_layer, n_class)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Checkpoints save and load functions
def save_checkpoint(checkpoint, filename='../checkpoint.pth.tar'):
    print('==> saving checkpoint')
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint):
    print('==> loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Train Network
def train(model):
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate((train_datloader)):

            # get data to cuda if possible
            data = data.to(device)
            # Squeeze the dim of data from (B X 1 X 28 X 28) to (B X 28 X 28)
            data = data.squeeze(dim=1)
            target = target.to(device)

            # feed forward the data to model
            scores = model(data)
            loss = criterion(scores, target)
            # print(f"epoch: {epoch}, batch: {batch_idx}, loss: {loss}")
            
            # Backpropagation
            optimizer.zero_grad() # This will flush the gradients from the last iteration
            loss.backward()

            # optimise the loss (gradient descent or Adam step)
            optimizer.step()

        # Create a model checkpoint
        if epoch % 3 == 0:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
            # print(f"epoch: {epoch}, batch: {batch_idx}, loss: {loss}")

# Check accuracy on traininf and test data (Validate model accuracy)
# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    """
    Check accuracy of our trained model given a loader and a model

    Parameters:
        loader: torch.utils.data.DataLoader
            A loader for the dataset you want to check accuracy on
        model: nn.Module
            The model you want to check accuracy on

    Returns:
        acc: float
            The accuracy of the model on the dataset given by the loader
    """

    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:

            # Move data to device
            x = x.to(device=device)
            # Squeeze the dim of x from (B X 1 X 28 X 28) to (B X 28 X 28)
            x = x.squeeze(dim=1)
            y = y.to(device=device)

            # Forward pass
            scores = model(x)
            predictions = scores.argmax(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# get model to cuda if possible
model = model.to(device)

# Train the model
print("Training model ...")
train(model)

# Load the model from checkpoint
if load_model:
    load_checkpoint(torch.load('../checkpoint.pth.tar'))

# Check accuracy on training & test to see how good our model
print('-'*100)
print(f"Accuracy on training set: {check_accuracy(train_datloader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_datloader, model)*100:.2f}")