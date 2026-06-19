# model.py
import torch
import torch.nn as nn
from collections import OrderedDict

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=20, num_classes=5):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def get_parameters(net):
    """Extracts PyTorch model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    """Loads a list of NumPy arrays back into a PyTorch model's state dict."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)