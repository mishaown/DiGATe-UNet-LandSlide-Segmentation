import os
from datetime import datetime

def init_logger(log_dir='training_logs', filename_prefix='training_report'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.txt")
    
    return log_file

def log_report(filepath, epoch, num_epochs, train_loss, train_dice, train_iou, val_loss, val_dice, val_iou, lr):
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(filepath, 'a') as f:
        f.write(f"[{time_str}] Epoch {epoch}/{num_epochs}\n")
        f.write(f"  Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}\n")
        f.write(f"  Val   Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}\n")
        f.write(f"  Learning Rate: {lr:.6f}\n")
        f.write("-" * 60 + "\n")