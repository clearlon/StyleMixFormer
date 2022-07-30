import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from styleformer.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)

@LOSS_REGISTRY.register()
class ArcFaceLoss(nn.Module):
    """ reference: <Additive Angular Margin Loss for Deep Face Recognition>
    """
    def __init__(self, feat_dim, num_class, s=64., m=0.5, loss_weight=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.loss_weight = loss_weight

        self.scale = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.theta = math.cos(math.pi - m)
        self.sinmm = math.sin(math.pi - m) * m
        self.easy_margin = False

        self.w = nn.Parameter(torch.Tensor(feat_dim, num_class))
        nn.init.xavier_normal_(self.w)

    def forward(self, x, labels):
        """
        x: embedding
        y: label
        """
        device = x.device
        with torch.no_grad():
            self.w.data = F.normalize(self.w.data, dim=0)
            
        cos_theta = F.normalize(x, dim=1).mm(self.w) # cos(theta)

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                cos_theta > 0, cos_theta_m, cos_theta)
        else:
            final_target_logit = torch.where(
                cos_theta > self.theta, cos_theta_m, cos_theta - self.sinmm)

        cos_theta = final_target_logit
        cos_theta = cos_theta * self.scale

        loss = F.cross_entropy(cos_theta, labels.long())

        return loss * self.loss_weight
