"""Optimizers for networks."""

import torch


def get_optimizer(parameters, cfg):
    """Get optimizer."""
    if cfg.optimizer == "adam":
        return torch.optim.Adam(
            parameters, lr=cfg.lr, betas=(cfg.beta1, 0.999))
    else:
        raise NotImplementedError("not implemented optimizer {}"
                                  .format(cfg.optimizer))
