"""Weight initializer for networks.

partly referenced from:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""

from functools import partial

from torch import nn


def init_layer_normal(layer):
    """Init layer weights with normal distribution."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.normal(layer.weight.data, 0.0, 0.02)
    elif layer_name.find("Linear") != -1:
        nn.init.normal(layer.weight.data, 0.0, 0.02)
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.normal(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_layer_uniform(layer):
    """Init layer weights with uniform distribution."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.uniform(layer.weight.data, 0.0, 1.0)
    elif layer_name.find("Linear") != -1:
        nn.init.uniform(layer.weight.data, 0.0, 1.0)
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.uniform(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_layer_xavier_normal(layer):
    """Init layer weights by xavier method with normal distribution."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.xavier_normal(layer.weight.data, gain=1)
    elif layer_name.find("Linear") != -1:
        nn.init.xavier_normal(layer.weight.data, gain=1)
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.uniform(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_layer_xavier_uniform(layer):
    """Init layer weights by xavier method with uniform distribution."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.xavier_uniform(layer.weight.data, gain=1)
    elif layer_name.find("Linear") != -1:
        nn.init.xavier_uniform(layer.weight.data, gain=1)
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.uniform(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_layer_kaiming_normal(layer):
    """Init layer weights by kaiming method with normal distribution."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.kaiming_normal(layer.weight.data, a=0, mode="fan_in")
    elif layer_name.find("Linear") != -1:
        nn.init.kaiming_normal(layer.weight.data, a=0, mode="fan_in")
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.uniform(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_layer_kaiming_uniform(layer):
    """Init layer weights by kaiming method with uniform distribution."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.kaiming_normal(layer.weight.data, a=0, mode="fan_in")
    elif layer_name.find("Linear") != -1:
        nn.init.kaiming_normal(layer.weight.data, a=0, mode="fan_in")
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.uniform(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_layer_orthogonal(layer):
    """Init layer weights with a (semi) orthogonal matrix."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.orthogonal(layer.weight.data, gain=1)
    elif layer_name.find("Linear") != -1:
        nn.init.orthogonal(layer.weight.data, gain=1)
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.uniform(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_layer_sparse(layer):
    """Init layer weights with a sparse matrix."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        nn.init.sparse(layer.weight.data, sparsity=0.1, std=0.01)
    elif layer_name.find("Linear") != -1:
        nn.init.sparse(layer.weight.data, sparsity=0.1, std=0.01)
    elif layer_name.find("BatchNorm2d") != -1:
        nn.init.uniform(layer.weight.data, 1.0, 0.02)
        nn.init.constant(layer.bias.data, 0.0)


def init_network_weights(net, init_type="normal"):
    """Init weights in the whole network."""
    print("initialize network by method: {}".format(init_type))
    if init_type == "normal":
        net.apply(init_layer_normal)
    elif init_type == "uniform":
        net.apply(init_layer_uniform)
    elif init_type == "xavier_normal":
        net.apply(init_layer_xavier_normal)
    elif init_type == "xavier_uniform":
        net.apply(init_layer_xavier_uniform)
    elif init_type == "kaiming_normal":
        net.apply(init_layer_kaiming_normal)
    elif init_type == "kaiming_uniform":
        net.apply(init_layer_kaiming_uniform)
    elif init_type == "orthogonal":
        net.apply(init_layer_orthogonal)
    elif init_type == "sparse":
        net.apply(init_layer_sparse)
    else:
        raise NotImplementedError(
            "not-implemented initialization method {}".format(init_type))


def get_initializer(init_type="normal"):
    """Get weight initializer."""
    return partial(init_network_weights, init_type=init_type)
