import torch

class BinarySegmentationMetrics:
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth = smooth

    def _prepare_tensors(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        pred_bin = (pred > self.threshold).float()
        target = target.float()

        pred_flat = pred_bin.view(-1)
        target_flat = target.view(-1)

        return pred_flat, target_flat

    def compute_classification_metrics(self, pred: torch.Tensor, target: torch.Tensor):
        pred_flat, target_flat = self._prepare_tensors(pred, target)

        TP = (pred_flat * target_flat).sum()
        TN = ((1 - pred_flat) * (1 - target_flat)).sum()
        FP = (pred_flat * (1 - target_flat)).sum()
        FN = ((1 - pred_flat) * target_flat).sum()

        accuracy = (TP + TN + self.smooth) / (TP + TN + FP + FN + self.smooth)
        precision = (TP + self.smooth) / (TP + FP + self.smooth)
        recall = (TP + self.smooth) / (TP + FN + self.smooth)
        f1 = (2 * precision * recall + self.smooth) / (precision + recall + self.smooth)

        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1.item()
        }

    def compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor):
        pred_flat, target_flat = self._prepare_tensors(pred, target)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return dice.item()

    def compute_iou_score(self, pred: torch.Tensor, target: torch.Tensor):
        pred_flat, target_flat = self._prepare_tensors(pred, target)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou.item()

    def compute_mean_iou(self, pred: torch.Tensor, target: torch.Tensor):
        # For binary segmentation, mean IoU = IoU
        return self.compute_iou_score(pred, target)
    