"""Helpful functions for models."""

import os

import torch


def init_weights(layer):
    """Init weights for layers."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)
    elif layer_name.find("Linear") != -1:
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()


def get_model(model, restore):
    """Get models with cuda and weights."""
    # init weights of model
    model.apply(init_weights)
    # restore model weights
    model = restore_model(model, restore)
    # check if cuda is available
    if torch.cuda.is_available():
        model.cuda()
    return model


def save_model(model, savedir, filename):
    """Save trained model."""
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    torch.save(model.state_dict(), os.path.join(savedir, filename))
    print("save model to {}".format(os.path.join(savedir, filename)))


def restore_model(model, restore):
    """Restore network from saved model."""
    if restore is not None and os.path.exists(restore):
        model.load_state_dict(torch.load(restore))
        model.restored = True
        print("restore model from {}".format(restore))
    return model
