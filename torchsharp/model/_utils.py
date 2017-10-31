"""Helpful functions for models."""

import torch
from torch.autograd import Variable


def print_network(net):
    """To print the architecture of network.

    Args:
        net (torch.nn.Module): Network object to be printed.
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print("Total number of parameters: {}".format(num_params))


def parse_variable(inputs: Variable) -> torch.LongTensor:
    """Parse Variable into Tenser.

    Args:
        inputs: input Variable

    Return:
        outputs: squeezed Tensor on cpu
    """
    if isinstance(inputs, Variable):
        return inputs.cpu().data.squeeze()
    else:
        raise TypeError("input instance is not Variable")
    return inputs
