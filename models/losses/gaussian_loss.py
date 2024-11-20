# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn as nn
from torch.distributions import Normal
from models.losses.utils import weighted_loss
from ..builder import LOSSES
from .utils import weight_reduce_loss


def gaussian_loss(pred, 
                  target, 
                  logstd, 
                  weight=None, 
                  reduction='mean',
                  avg_factor=None):
    """Smooth L1 loss with uncertainty.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        sigma (torch.Tensor): The sigma for uncertainty.
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() == logstd.size(), 'The size of pred ' \
        f'{pred.size()}, target {target.size()}, and logstd {logstd.size()} ' \
        'are inconsistent.'
    dist = Normal(pred, torch.exp(logstd))
    likelihood = dist.log_prob(target)
    if weight is not None:
        likelihood = likelihood * weight + 0.2*logstd

    loss_likelihood = weight_reduce_loss(likelihood, weight, reduction, avg_factor)
    return -loss_likelihood

@LOSSES.register_module()
class GaussianLoss(nn.Module):
    r"""Smooth L1 loss with uncertainty.

    Please refer to `PGD <https://arxiv.org/abs/2107.14160>`_ and
    `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
    and Semantics <https://arxiv.org/abs/1705.07115>`_ for more details.

    Args:
        alpha (float, optional): The coefficient of log(sigma).
            Defaults to 1.0.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(GaussianLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                logstd,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            sigma (torch.Tensor): The sigma for uncertainty.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * gaussian_loss(
            pred,
            target,
            logstd,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox