import torch
import segmentation_models_pytorch as smp

def get_statistics(output: torch.Tensor, target: torch.Tensor, mode: str = 'binary', threshold: float = 0.5):
    """
    Computes true positives, false positives, false negatives, and true negatives.

    Args:
        output (torch.Tensor): The prediction tensor.
        target (torch.Tensor): The ground truth tensor.
        mode (str): The mode to use in SMP metrics (default: 'binary').
        threshold (float): Threshold for binarizing predictions (default: 0.5).

    Returns:
        Tuple[tp, fp, fn, tn]: Statistics tensors.
    """
    return smp.metrics.get_stats(output, target, mode=mode, threshold=threshold)

def iou(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction)

def f1(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction)

def f2(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    return smp.metrics.fbeta_score(tp, fp, fn, tn, beta=2, reduction=reduction)

def acc(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    return smp.metrics.accuracy(tp, fp, fn, tn, reduction=reduction)

def recall(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    return smp.metrics.recall(tp, fp, fn, tn, reduction=reduction)

def prec(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    return smp.metrics.precision(tp, fp, fn, tn, reduction=reduction)


def compute_f1_score(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, reduction: str = "micro-imagewise"):
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='binary', threshold=threshold)
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction)

def compute_dice_score(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, reduction: str = "micro-imagewise"):
    # Dice = F1
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='binary', threshold=threshold)
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction=reduction)

def compute_iou_score(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, reduction: str = "micro-imagewise"):
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode='binary', threshold=threshold)
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction=reduction)
