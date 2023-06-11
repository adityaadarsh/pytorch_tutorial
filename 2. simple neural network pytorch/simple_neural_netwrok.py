# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transformations
import matplotlib.pyplot as plt

# Create a Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, n_classes):
        super(NN, self).__init__()

        self.fc1= nn.Linear(input_size, 50)
        self.fc2= nn.Linear(50, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# HyperParameters
input_dim = 784
n_class = 10
learning_rate = 0.001
batch_size = 32
num_epochs = 1

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
model = NN(input_dim, n_class)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network

# get model to cuda if possible
model = model.to(device)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate((train_datloader)):

        # get data to cuda if possible
        data = data.to(device)
        target = target.to(device)

        data = data.reshape(-1,784)

        # feed forward the data to model
        scores = model(data)
        loss = criterion(scores, target)
        print(f"epoch: {epoch}, batch: {batch_idx}, loss: {loss}")
        
        # Backpropagation
        optimizer.zero_grad() # This will flush the gradients from the last iteration
        loss.backward()

        # optimise the loss (gradient descent or Adam step)
        optimizer.step()

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
            y = y.to(device=device)

            # Get to correct shape
            x = x.reshape(x.shape[0], -1)

            # Forward pass
            scores = model(x)
            predictions = scores.argmax(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples

# Check accuracy on training & test to see how good our model
print('-'*100)
print(f"Accuracy on training set: {check_accuracy(train_datloader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_datloader, model)*100:.2f}")