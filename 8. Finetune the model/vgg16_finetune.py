
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

# Load pretrain model & modify it
model = torchvision.models.vgg16(weights="DEFAULT", progress=True)

# set requires_grad = False for the neck of model to finetune

for param in model.parameters():
    param.requires_grad = False

# If want to train whole model with pre-trained weights
# Then keep param.requires_grad = True or delete above two lines

# Modify the neck of model according to the use case
model.avgpool = nn.Identity()
model.classifier = nn.Sequential(
    nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, num_classes)
)
model.to(device)


# Load CIFAR data
train_dataset = datasets.CIFAR10(
    root="8. Finetune the model/dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train network
from tqdm import tqdm
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")

    
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0

    # Eval mode
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    # Back to train mode
    model.train()

# Check accuracy
check_accuracy(train_loader, model)

