import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2 

EPSILON = 1e-6

class SingleComposite_l4s(Dataset):
    def __init__(self, dataset, bands='14Bands', resize_to=None, transform=None):
        self.dataset = dataset
        self.bands = bands
        self.transform = transform
        if isinstance(resize_to, int):
            self.resize_to = (resize_to, resize_to)
        else:
            self.resize_to = resize_to

        # mean and std for normalization
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        
        # B2:Blue, B3:Green, B4:Red, B8:NIR, B11:SWIR, B13:Slope, B14:Elevation(DEM)
        self.band_map = {
            'B1': 0, 'B2': 1, 'B3': 2, 'B4': 3,
            'B5': 4, 'B6': 5, 'B7': 6, 'B8': 7,
            'B9': 8, 'B10': 9, 'B11': 10, 'B12': 11,
            'B13': 12, 'B14': 13 }
    
    def __create_composite(self, image):
        
        bands = {f'B{i}': image[self.band_map[f'B{i}']] for i in range(1, 15)} # Access like: # B1 = bands['B1']

        if self.bands == "RGB":
            return np.stack([bands['B4'], bands['B3'], bands['B2']], axis=0)
        if self.bands == "14Bands":
            return np.stack([bands[f'B{i}'] for i in range(1, 15)], axis=0)
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
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, filename = self.dataset[idx]

        # Apply standardization to the image
        # for i in range(len(self.mean)):
        #     image[i, :, :] = (image[i, :, :] - self.mean[i]) / (self.std[i] + EPSILON) # Added EPSILON to std to prevent division by zero
        
        # Creating composites
        comp = self.__create_composite(image)

        # Resize
        if self.resize_to:
            comp = self.__resize(comp)
            label = cv2.resize(label, self.resize_to, interpolation=cv2.INTER_NEAREST)

        # Convert to tensor
        comp_tensor = self.__normalize(torch.from_numpy(comp).float())
        label_tensor = torch.from_numpy(label).float() 

        # Transformation
        if self.transform:
            comp_tensor, label_tensor = self.transform(comp_tensor, label_tensor)
        
        return comp_tensor, label_tensor