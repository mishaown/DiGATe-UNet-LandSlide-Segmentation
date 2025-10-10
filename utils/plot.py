import os
import matplotlib.pyplot as plt

def plot_training_metrics(history, ex=None):
    epochs = range(1, len(history['train_loss']) + 1)

    # Create a new figure with subplots
    plt.figure(figsize=(18, 6))

    # Plot training and validation losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    plt.plot(epochs, history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation Dice coefficients
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_dice'], label='Train Dice Coeff', color='green')
    plt.plot(epochs, history['val_dice'], label='Validation Dice Coeff', color='red')
    plt.title('Training and Validation Dice Coefficients')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    # Plot training and validation IoU
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_iou'], label='Train IoU', color='purple')
    plt.plot(epochs, history['val_iou'], label='Validation IoU', color='brown')
    plt.title('Training and Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()

    output_path = f'visuals/{ex}/training_loop.png'
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


