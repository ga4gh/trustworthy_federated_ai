import torch
import torch.nn as nn
from collections import OrderedDict

# Unified superpopulations mapping for the 5 global ancestry classes
SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]

class AncestryNet(nn.Module):
    def __init__(self, input_dim=10, num_classes=5):  # 10 principal components
        super(AncestryNet, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.fc_layer(x)

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)