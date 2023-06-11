import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transformations
import torch.onnx as onnx
import onnxruntime

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

# Load the model 
# Parameters for network initialisation
n_class = 10
n_layer = 10
hidden_size = 5
emb_dim = 28
batch_size = 64

# Load model architecture
model = BLSTM(emb_dim, hidden_size, n_layer, n_class)
# Use the gpu if possible
model = model.to(device)

# Load checkpoint
checkpoint = torch.load('../checkpoint.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# Note: Be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.
model.eval()

# Export the model using onnx
onnx_model = './model.onnx'
input_image = torch.zeros((1,28,28)).to(device)
onnx.export(model, input_image, onnx_model, verbose = False)


max_seq_length = 12

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
# test dataloader
test_dataset = datasets.MNIST(root='dataset', train=False, transform=transformations.ToTensor(), download=True)
test_datloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


session = onnxruntime.InferenceSession(onnx_model, None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load a test datapoint
x, y = test_dataset[0]

# Get class label list
classes = test_dataset.classes

# Prediction
print('*'*100)
print('*'*100)
result = session.run([output_name], {input_name: x.numpy()})
predicted, actual = classes[result[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')

# End