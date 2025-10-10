import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2 

EPSILON = 1e-6

class TwoComposites(Dataset):
    def __init__(self, dataset, bands='RGB&DEM', resize_to=None, transform=None):
        self.dataset = dataset
        self.bands = bands
        self.transform = transform
        if isinstance(resize_to, int):
            self.resize_to = (resize_to, resize_to)
        else:
            self.resize_to = resize_to

    def __create_composite(self, image, dem):
        # Inputs are (C, H, W), convert back to (H, W, C) for channel-wise slicing
        image = np.transpose(image, (1, 2, 0))
        dem = np.transpose(dem, (1, 2, 0))

        C1 = image[:, :, 0:1]
        C2 = image[:, :, 1:2]
        C3 = image[:, :, 2:3]
        C4 = dem[:, :, 0:1]

        if self.bands == "RGB&DEM":
            # (H, W, 3) then transpose to (3, H, W)
            comp1 = np.concatenate([C1, C2, C3], axis=-1)
            comp2 = np.concatenate([C4, C4, C4], axis=-1)

            comp1 = np.transpose(comp1, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            comp2 = np.transpose(comp2, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            return comp1, comp2
        else: 
            raise ValueError(f"Composite Error: '{self.bands}' ❌")

    
    def __normalize(self, image_tensor):
        # Normalize each channel to [0, 1] with np.clip
        for i in range(image_tensor.shape[0]):
            channel = image_tensor[i]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                channel = (channel - min_val) / (max_val - min_val + EPSILON)
            image_tensor[i] = torch.clamp(channel, 0.0, 1.0)
        return image_tensor
    
    def __resize(self, img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # (H, W) -> (H, W, 1)
        # Resize using cv2
        resized = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)
        # If original had 1 channel, re-expand it after resize
        if resized.ndim == 2:
            resized = np.expand_dims(resized, axis=-1)
        resized = np.transpose(resized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        return resized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image, dem, label = sample['image'], sample['dem'], sample['mask']

        # Resize
        if self.resize_to:
            image = self.__resize(image)
            dem = self.__resize(dem)
            label = cv2.resize(label, self.resize_to, interpolation=cv2.INTER_NEAREST)

        # Creating composites
        comp1, comp2 = self.__create_composite(image, dem)

        comp1_tensor = self.__normalize(torch.from_numpy(comp1).float())
        comp2_tensor = self.__normalize(torch.from_numpy(comp2).float())
        
        label_tensor = torch.from_numpy(label).float()
        label_tensor = (label_tensor > 0).float()


        # Transformation
        if self.transform:
            comp1_tensor, comp2_tensor, label_tensor = self.transform(comp1_tensor, comp2_tensor, label_tensor)
        
        return comp1_tensor, comp2_tensor, label_tensor