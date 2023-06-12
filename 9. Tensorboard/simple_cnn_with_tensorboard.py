# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transformations
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision

# Create a Simple CNN Network
class SimpleCnn(nn.Module):
    def __init__(self, input_channel, output_class):
        super(SimpleCnn, self).__init__()

        # CNN layers
        self.conv_layer_1 = nn.Conv2d(
            in_channels=input_channel,
            out_channels=8,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
        )

        self.pool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv_layer_2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
        )
    
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=output_class)

    def forward(self, x):
        x = F.relu(self.conv_layer_1(x))
        x = self.pool_layer(x)
        x = F.relu(self.conv_layer_2(x))
        x = self.pool_layer(x)
        x = x.flatten(1,-1)
        x = self.fc1(x)

        return x
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# HyperParameters
input_channel = 1
n_class = 10
learning_rates = [0.1]
batch_size = [16, 64]
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(
    root="/dataset/", train=True, transform=transformations.ToTensor(), download=True
)

test_dataset = datasets.MNIST(
    root="/dataset/", train=False, transform=transformations.ToTensor(), download=True
)

# Class label
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Training and logging parameters into the tensorboard
for batch_size in batch_size:
    for learning_rate in learning_rates:
        step = 0
        # Initialize network
        model = SimpleCnn(input_channel=input_channel, output_class=n_class)
        model.to(device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        writer = SummaryWriter(
            f"9. Tensorboard/runs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}"
        )

        # Visualize model in TensorBoard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(device))
        writer.close()

        for epoch in range(num_epochs):
            losses = []
            accuracies = []

            for batch_idx, (data, targets) in enumerate(train_loader):
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
                optimizer.step()

                # Calculate 'running' training accuracy
                features = data.reshape(data.shape[0], -1)
                img_grid = torchvision.utils.make_grid(data)
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                # Plot things to tensorboard
                class_labels = [classes[label] for label in predictions]
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("fc1", model.fc1.weight)
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar(
                    "Training Accuracy", running_train_acc, global_step=step
                )

                if batch_idx == 230:
                    writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=data,
                        global_step=batch_idx,
                    )
                step += 1

            writer.add_hparams(
                {"lr": learning_rate, "bsize": batch_size},
                {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "loss": sum(losses) / len(losses),
                },
            )