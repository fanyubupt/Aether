import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    """Initializes the weights and biases of a given module."""

    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    """Generate N deep copies of a given module and return them as a ModuleList."""

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    """Converts the input to a PyTorch tensor if it is a NumPy array."""

    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output