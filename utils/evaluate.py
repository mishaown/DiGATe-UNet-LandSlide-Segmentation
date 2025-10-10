import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Assuming smp_metrics is in models/smp_metrics.py
import models.smp_metrics as sm

def prep_batch(x1, x2, y, device):
    """Prepares a batch of data for the model."""
    x1 = x1.float().to(device, non_blocking=True)
    x2 = x2.float().to(device, non_blocking=True)
    y = y.long().to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x1, x2, y

def evaluate_model(model, dataset, device, set):
    """
    Evaluates the model on the given dataset and computes average metrics.
    """
    data_loader = DataLoader(dataset, shuffle=False, batch_size=32)
    model.to(device).eval()
    
    # Store metrics for each batch in lists
    metrics = {'acc': [], 'recall': [], 'prec': [], 'f1': [], 'iou': []}

    with torch.no_grad():
        for x1, x2, y in tqdm(data_loader, desc="Evaluating"):
            x1, x2, y = prep_batch(x1, x2, y, device)
            
            # Handle model outputs without auxiliary outputs
            out = model(x1, x2)
            y_main = out[0] if isinstance(out, (tuple, list)) else out

            # Calculate statistics for the batch
            tp, fp, fn, tn = sm.get_statistics(y_main, y, mode='binary', threshold=0.5)

            metrics['acc'].append(sm.acc(tp, fp, fn, tn).item())
            metrics['recall'].append(sm.recall(tp, fp, fn, tn).item())
            metrics['prec'].append(sm.prec(tp, fp, fn, tn).item())
            metrics['f1'].append(sm.f1(tp, fp, fn, tn).item())
            metrics['iou'].append(sm.iou(tp, fp, fn, tn).item())

    # Compute the average of each metric across all batches
    avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
    
    print(f"\n--- Evaluation Metrics on {set} Set---")
    for key, value in avg_metrics.items():
        print(f"{key.capitalize():<10}: {value:.4f}")
    print("--------------------------")
    
    # return avg_metrics

def viz_pred(model, dataset, device, ex_no, num_samples=5):
    """
    Visualizes model predictions against ground truth for a number of random samples.
    """
    model.to(device).eval()
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    fig.suptitle("Model Predictions vs. Ground Truth", fontsize=16)

    for i, idx in enumerate(sample_indices):
        x1, x2, y_true = dataset[idx] 

        # Add batch dimension (B, C, H, W) and move to device
        x1_dev = x1.unsqueeze(0).to(device)
        x2_dev = x2.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x1_dev, x2_dev)
            y_main = out[0] if isinstance(out, (tuple, list)) else out
            y_pred = (torch.sigmoid(y_main) > 0.5).float()

        # Prepare tensors for plotting (move to CPU and convert to NumPy)
        image_np = x1.permute(1, 2, 0).cpu().numpy()
        true_mask_np = y_true.squeeze().cpu().numpy()
        pred_mask_np = y_pred.squeeze().cpu().numpy()

        # Plot Original Image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f"Image #{idx}" if i == 0 else "")
        axes[i, 0].axis("off")
        
        # Plot Ground Truth Mask
        axes[i, 1].imshow(true_mask_np, cmap="viridis")
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis("off")
        
        # Plot Predicted Mask
        axes[i, 2].imshow(pred_mask_np, cmap="viridis")
        axes[i, 2].set_title("Prediction" if i == 0 else "")
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make space for suptitle

    # Create directory and save the figure
    output_path = f'visuals/{ex_no}/pred_samples.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\nSaved visualization to {output_path}")
    plt.show()