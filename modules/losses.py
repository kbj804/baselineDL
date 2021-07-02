"""
"""

import torch.nn as nn

def get_loss_fn(loss_fn_str: str):
    """ Loss 함수 반환하는 함수

    Returns:
        loss_fn (Callable)
    """

    if loss_fn_str == 'mseloss':
        return nn.MSELoss

