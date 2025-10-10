import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

class SegmentationLosses:
    def __init__(self, loss_name='dice', mode='binary', alpha=None, beta=None):
        """
        Initialize the loss function from SMP by name.
        
        Args:
            loss_name (str): one of ['dice', 'focal', 'tversky', 'lovasz', 'soft_bce']
            mode (str): 'binary', 'multiclass', or 'multilabel'
        """
        self.loss_name = loss_name.lower()
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
                
        if self.loss_name == 'dice':
            self.loss_fn = smp.losses.DiceLoss(mode=mode)
        elif self.loss_name == 'focal':
            self.loss_fn = smp.losses.FocalLoss(mode=mode)
        elif self.loss_name == 'tversky':
            self.loss_fn = smp.losses.TverskyLoss(mode=mode, alpha=alpha, beta=beta)
        elif self.loss_name == 'lovasz':
            # LovaszLoss needs probabilities as input
            self.loss_fn = smp.losses.LovaszLoss(mode=mode)
        elif self.loss_name == 'soft_bce':
            self.loss_fn = smp.losses.SoftBCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

    def __call__(self, outputs, targets):
        """
        Compute the loss given model outputs and targets.

        Args:
            outputs (torch.Tensor): raw logits from model
            targets (torch.Tensor): ground truth masks
        
        Returns:
            torch.Tensor: computed loss
        """
        if self.loss_name == 'lovasz':
            # LovaszLoss expects probabilities, so apply sigmoid for binary mode
            if self.mode == 'binary':
                probs = torch.sigmoid(outputs)
            elif self.mode == 'multiclass':
                probs = torch.softmax(outputs, dim=1)
            else:
                raise NotImplementedError("LovaszLoss currently supports only 'binary' or 'multiclass' mode.")
            
            loss = self.loss_fn(probs, targets)
        else:
            # Other losses take raw logits directly
            loss = self.loss_fn(outputs, targets)

        return loss