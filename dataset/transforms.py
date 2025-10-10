import torchvision.transforms.functional as TF
import random
import torch
import numpy as np
import cv2

class SingleStreamTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = TF.hflip(image)
            label = TF.hflip(label)

        if random.random() < self.p:
            image = TF.vflip(image)
            label = TF.vflip(label)

        if random.random() < self.p:
            image = add_gaussian_noise(image)

        if random.random() < self.p:
            image = add_salt_pepper_noise(image)

        if random.random() < self.p:
            image = apply_clahe(image)

        return image, label

class DualStreamTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image1, image2, label):
        if random.random() < self.p:
            image1 = TF.hflip(image1)
            image2 = TF.hflip(image2)
            label = TF.hflip(label)

        if random.random() < self.p:
            image1 = TF.vflip(image1)
            image2 = TF.vflip(image2)
            label = TF.vflip(label)

        if random.random() < self.p:
            image1 = add_gaussian_noise(image1)
            image2 = add_gaussian_noise(image2)

        if random.random() < self.p:
            image1 = add_salt_pepper_noise(image1)
            image2 = add_salt_pepper_noise(image2)

        if random.random() < self.p:
            image1 = apply_clahe(image1)
            image2 = apply_clahe(image2)

        return image1, image2, label


def add_gaussian_noise(tensor_img, mean=0.0, std=0.05):
    noise = torch.randn_like(tensor_img) * std + mean
    return tensor_img + noise


def add_salt_pepper_noise(tensor_img, amount=0.005, salt_vs_pepper=0.5):
    img = tensor_img.clone().detach().cpu().numpy()
    c, h, w = img.shape
    num_pixels = int(amount * h * w)

    for i in range(c):
        # Salt
        coords = [np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels)]
        img[i][tuple(coords)] = 1.0
        # Pepper
        coords = [np.random.randint(0, h, num_pixels), np.random.randint(0, w, num_pixels)]
        img[i][tuple(coords)] = 0.0

    return torch.from_numpy(img).type_as(tensor_img)


def apply_clahe(tensor_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = tensor_img.clone().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    img = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    for i in range(img.shape[2]):
        img[:, :, i] = clahe.apply(img[:, :, i])

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    return torch.from_numpy(img).type_as(tensor_img)