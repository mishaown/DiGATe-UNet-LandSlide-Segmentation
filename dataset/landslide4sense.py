import numpy as np
from torch.utils.data import Dataset
import h5py
import os
import glob

EPSILON = 1e-6

class LandSlide4Sense(Dataset):
    def __init__(self, data_dir):
        self.images = sorted(glob.glob(os.path.join(data_dir, "img/*.h5")))
        self.masks = sorted(glob.glob(os.path.join(data_dir, "mask/*.h5")))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        file_name = os.path.basename(img_path)

        with h5py.File(img_path, 'r') as hdf:
            img = np.array(hdf.get('img'), np.float32)
            img[np.isnan(img)] = EPSILON 
            img = img.transpose((-1, 0, 1)) # HxWxC -> CxHxW

        with h5py.File(mask_path, 'r') as hdf:
            mask = np.array(hdf.get('mask'), np.float32)

        return img, mask, file_name