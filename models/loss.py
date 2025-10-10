import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice_score = (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        # pred = torch.sigmoid(pred)
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pred = pred.view(-1)
        target = target.view(-1)

        tp = (pred * target).sum()
        fp = ((1 - target) * pred).sum()
        fn = ((1 - pred) * target).sum()

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky_index

def weighted_bce_loss(pred, target, pos_weight):
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

def dice_bce_loss(pred, target, bce_weight=1.0):
    dice = DiceLoss()(pred, target)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return dice + bce_weight * bce

def dice_focal_loss(pred, target, focal_weight=1.0):
    dice = DiceLoss()(pred, target)
    focal = FocalLoss()(pred, target)
    return dice + focal_weight * focal

def combined_loss(pred, target, alpha=1.0, beta=1.0, pos_weight=5.0):
    pos_weight_tensor = torch.tensor(pos_weight, dtype=pred.dtype, device=pred.device)
    dice = DiceLoss()(pred, target)
    focal = FocalLoss()(pred, target)
    bce = weighted_bce_loss(pred, target, pos_weight=pos_weight_tensor)
    return dice + alpha * focal + beta * bce

def tversky_bce_loss(pred, target, bce_weight=0.5):
    tversky = TverskyLoss()(pred, target)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return tversky + bce_weight * bce

