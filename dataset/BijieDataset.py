import os, numpy as np
from PIL import Image
from torch.utils.data import Dataset

def load_image(fp): return np.array(Image.open(fp))

class BijieRawDataset(Dataset):
    """
    Just loads (image,dem,mask) as numpy arrays, no transforms, no origin.
    """
    def __init__(self, root_dir, phase="landslide"):
        self.image_dir = os.path.join(root_dir, "image")
        self.dem_dir   = os.path.join(root_dir, "dem")
        self.mask_dir  = None if phase=="non-landslide" else os.path.join(root_dir, "mask")
        self.files     = sorted(f for f in os.listdir(self.image_dir) if f.endswith(".png"))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        fn = self.files[i]
        img = load_image(os.path.join(self.image_dir, fn))
        dem = load_image(os.path.join(self.dem_dir,   fn))
        if self.mask_dir:
            mask = load_image(os.path.join(self.mask_dir, fn))
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        return {"image":img, "dem":dem, "mask":mask}