"""Helpful functions for models."""


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
