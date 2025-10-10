import numpy as np
import matplotlib.pyplot as plt
import os
import torch

titles = ["RGB", "NDVI", "SLOPE", "ELEVATION", "MASKS"]
cmaps = [None, 'viridis', 'terrain', 'terrain', 'viridis']

def show14bands(dataset, save_path=None):
    num_of_samples = 3
    sample_indices = np.random.choice(len(dataset), size=num_of_samples, replace=False)
    print("Sample indices:", sample_indices)
    
    for idx in sample_indices:
        sample = dataset[idx]
        
        # Unpack sample with fallback if dataset is unlabeled
        if len(sample) == 4:
            image, label, size, name = sample
        elif len(sample) == 3:  # unlabeled case
            image, size, name = sample
            label = None
        else:
            raise ValueError("Unexpected number of elements returned from dataset")

        num_bands = image.shape[0]
        
        ncols = num_bands + (1 if label is not None else 0)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        
        if ncols == 1:
            axes = [axes]
        
        # Plot all image bands
        for i in range(num_bands):
            ax = axes[i]
            ax.imshow(image[i], cmap='terrain')
            ax.set_title(f"Band {i}")
            ax.axis('off')
        
        # Plot label in last column if exists
        if label is not None:
            ax = axes[-1]
            ax.imshow(label, cmap='viridis')
            ax.set_title("Label")
            ax.axis('off')
        
        plt.suptitle(f"Sample: {name}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

def after_transform(dataset, ex, titles, cmaps):
    num_of_samples = 3
    total_samples = len(dataset)
    sample_indices = np.random.randint(1, total_samples, size=3)

    fig, axes = plt.subplots(num_of_samples, 5, figsize=(15, 4 * num_of_samples))

    for r_idx, i_idx in enumerate(sample_indices):
        image1, image2, mask  = dataset[i_idx]

        image1 = np.transpose(image1, (1, 2, 0)) # CxHxW -> HxWxC
        image2 = np.transpose(image2, (1, 2, 0)) # CxHxW -> HxWxC

        rgb = image1[:, :, 0:3]
        ndvi = image2[:, :, 0:1]
        slope = image2[:, :, 1:2]
        dem = image2[:, :, 2:3]

        for c_idx, (data, title, cmap) in enumerate(zip([rgb, ndvi, slope, dem, mask], titles, cmaps)):
            ax = axes[r_idx, c_idx]
            if title == "RGB":
                ax.imshow(data)
            else:
                im = ax.imshow(data, cmap=cmap)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(f"{title} ({i_idx})")
            ax.axis('off')

    output_path = f'visuals/{ex}/dataset_sample.png'
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.tight_layout()
    plt.show()
