"""Scheduler for learning rate."""

from torch.optim import lr_scheduler


def get_scheduler(optimizer, cfg):
    """Get lr scheduler."""
    if cfg.lr_policy == "lambda":
        def lambda_rule(epoch):
            return 1.0 - (epoch + 1) / cfg.lr_decay_epoch[0]
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=cfg.lr_decay_epoch[0],
                                        gamma=cfg.lr_decay_factor)
    elif cfg.lr_policy == "multistep":
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=cfg.lr_decay_epoch,
                                             gamma=cfg.lr_decay_factor)
    elif cfg.lr_policy == "exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer,
                                               gamma=cfg.lr_decay_factor,
                                               last_epoch=-1)
    elif cfg.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode="min",
                                                   factor=cfg.lr_decay_factor,
                                                   threshold=0.01,
                                                   patience=10)
    else:
        return NotImplementedError("not implemented learning rate policy {}"
                                   .format(cfg.lr_policy))
    return scheduler
