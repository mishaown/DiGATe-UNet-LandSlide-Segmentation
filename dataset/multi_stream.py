import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import cv2 

EPSILON = 1e-6

class TwoComposites_l4s(Dataset):
    def __init__(self, dataset, bands='RGB&FCIR', resize_to=None, transform=None):
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
        
        # B2:Blue, B3:Green, B4:Red, B8:NIR, B8A: Narrow NIR, B11:SWIR 1, B12: SWIR 2, B13:Slope, B14:Elevation(DEM)
        self.band_map = {'B2': 1, 'B3': 2, 'B4': 3, 'B8': 7, 'B8A':8, 'B11': 10, 'B12': 11, 'B13': 12, 'B14': 13}
    
    def __create_composite(self, image):
        B2 = image[self.band_map['B2']]
        B3 = image[self.band_map['B3']]
        B4 = image[self.band_map['B4']]
        B8 = image[self.band_map['B8']]
        B8A = image[self.band_map['B8A']]
        B11 = image[self.band_map['B11']]
        B12 = image[self.band_map['B12']]
        B13 = image[self.band_map['B13']]
        B14 = image[self.band_map['B14']]

        if self.bands == "RGB&FCIR":
            # Stream 1: RGB & Stream 2: False Color Infrared
            comp1 = np.stack([B4, B3, B2], axis=0)
            comp2 = np.stack([B8, B4, B11], axis=0)
            return comp1, comp2
        elif self.bands == "RGB&NGB":
            # Stream 1: RGB & Stream 2: NGB - NIR, GREEN, BLUE
            comp1 = np.stack([B4, B3, B2], axis=0)
            comp2 = np.stack([B8, B3, B2], axis=0)
            return comp1, comp2
        
        elif self.bands == "RGB&SWIR":
            # Stream 1: RGB & Stream 2: SWIR (Short-Wave IR), SWIR, NIR
            comp1 = np.stack([B4, B3, B2], axis=0)
            comp2 = np.stack([B11, B12, B8A], axis=0)
            return comp1, comp2
        
        elif self.bands == "RGB&DEM":
            # Stream 1: RGB, Stream 2: DEM
            comp1 = np.stack([B4, B3, B2], axis=0)
            comp2 = np.stack([B14, B14, B14], axis=0) # DEM
            return comp1, comp2 
        elif self.bands == "RGB-Topo":
            # Stream 1: RGB, Stream 2: Topography 
            comp1 = np.stack([B4, B3, B2], axis=0)
            comp2 = np.stack([B14, B13, B2], axis=0) # DEM, Slope, Blue
            return comp1, comp2 
        elif self.bands == "RGB-NSE":
            # Stream 1: RGB & Stream 2: NDVI, Slope, Dem
            comp1 = np.stack([B4, B3, B2], axis=0)
            ndvi = (B8 - B4) / (B8 + B4 + EPSILON)
            comp2 = np.stack([ndvi, B13, B14], axis=0)
            return comp1, comp2
        elif self.bands == "NDVI-RGB":
            # Stream 1: NDVI grayscale repeated 3 times, Stream 2: RGB
            ndvi = np.clip((B8 - B4) / (B8 + B4 + EPSILON), -1, 1)
            comp1 = np.stack([ndvi, ndvi, ndvi], axis=0)
            comp2 = np.stack([B4, B3, B2], axis=0)
            return comp1, comp2
        
        elif self.bands == "RGB-NDVI-SLOPE-DEM":
            # Stream 1: RGB, Stream 2: NDVI grayscale repeated 3 times
            ndvi = np.clip((B8 - B4) / (B8 + B4 + EPSILON), -1, 1)
            comp1 = np.stack([B4, B3, B2], axis=0)
            comp2 = np.stack([ndvi, B13, B14], axis=0)

            return comp1, comp2 
        elif self.bands == "NDVI-Topo":
            # Stream 1: NDVI grayscale repeated 3 times, Stream 2: Topographic
            ndvi = (B8 - B4) / (B8 + B4 + EPSILON)
            comp1 = np.stack([ndvi, ndvi, ndvi], axis=0)
            comp2 = np.stack([B14, B13, B2], axis=0)
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
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, filename = self.dataset[idx]

        '''
        image.shape -> 14,128,128, type -> numpy.ndarray
        label.shape -> 128,128, type -> numpy.ndarray
        Task:
            1. Select Bands and Calculate Indices
            2. Resize Images, Labels and normalize each channels
            3. Apply Transformations (Augmentation)
        '''
        # Apply standardization to the image
        # for i in range(len(self.mean)):
        #     image[i, :, :] = (image[i, :, :] - self.mean[i]) / (self.std[i] + EPSILON) # Added EPSILON to std to prevent division by zero
        
        # Creating composites
        comp1, comp2 = self.__create_composite(image)

        # Resize
        if self.resize_to:
            comp1 = self.__resize(comp1)
            comp2 = self.__resize(comp2)
            label = cv2.resize(label, self.resize_to, interpolation=cv2.INTER_NEAREST)

        comp1_tensor = self.__normalize(torch.from_numpy(comp1).float())
        comp2_tensor = self.__normalize(torch.from_numpy(comp2).float())
        label_tensor = torch.from_numpy(label).float() 

        # Convert to tensor
        # comp1_tensor = torch.from_numpy(comp1).float()
        # comp2_tensor = torch.from_numpy(comp2).float()
        # label_tensor = torch.from_numpy(label).float()

        # Transformation
        if self.transform:
            comp1_tensor, comp2_tensor, label_tensor = self.transform(comp1_tensor, comp2_tensor, label_tensor)
        
        return comp1_tensor, comp2_tensor, label_tensor