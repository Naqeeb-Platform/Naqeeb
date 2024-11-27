import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import cv2
from data.data_utils import perlin_noise

class MVTecDataset(Dataset):
    def __init__(  # Changed from 'init' to '__init__'
        self,
        is_train,
        mvtec_dir,
        resize_shape=[256, 256],
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        dtd_dir=None,
        rotate_90=False,
        random_rotate=0,
        additional_augmentation=False,
    ):
        super().__init__()  # Corrected to call the superclass constructor
        self.resize_shape = resize_shape
        self.is_train = is_train
        if is_train:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*.png") + 
                                      glob.glob(mvtec_dir + "/*.jpeg") + glob.glob(mvtec_dir + "/*.JPG")+glob.glob(mvtec_dir + "/*.HEIC")+glob.glob(mvtec_dir + "/*.heic")+
                                      glob.glob(mvtec_dir + "/*.jpg"))
            self.dtd_paths = sorted(glob.glob(dtd_dir + "/*/*.jpg"))
            self.rotate_90 = rotate_90
            self.random_rotate = random_rotate
            self.additional_augmentations = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5)
            ])
        else:
            self.mvtec_paths = sorted(glob.glob(mvtec_dir + "/*/*.png") + 
                                      glob.glob(mvtec_dir + "/*/*.jpeg") + 
                                      glob.glob(mvtec_dir + "/*/*.jpg")+glob.glob(mvtec_dir + "/*.JPG"))
        
        self.final_preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )

    def __len__(self):  # Changed from 'len' to '__len__'
        return len(self.mvtec_paths)

    def __getitem__(self, index):  # Changed from 'getitem' to '__getitem__'
        image = Image.open(self.mvtec_paths[index]).convert("RGB")
        image = image.resize(self.resize_shape, Image.Resampling.BILINEAR)

        if self.is_train:
            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize(self.resize_shape, Image.Resampling.BILINEAR)

            fill_color = (114, 114, 114)
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(degree, fillcolor=fill_color, resample=Image.Resampling.BILINEAR)
            if self.random_rotate > 0:
                degree = np.random.uniform(-self.random_rotate, self.random_rotate)
                image = image.rotate(degree, fillcolor=fill_color, resample=Image.Resampling.BILINEAR)

            aug_image, aug_mask = perlin_noise(image, dtd_image, aug_prob=1.0)

            aug_image = self.final_preprocessing(aug_image)
            image = self.final_preprocessing(image)

            return {"img_aug": aug_image, "img_origin": image, "mask": aug_mask}
        else:
            image = self.final_preprocessing(image)
            dir_path, file_name = os.path.split(self.mvtec_paths[index])
            base_dir = os.path.basename(dir_path)
            if base_dir == "good":
                mask = torch.zeros_like(image[:1])
            else:
                base_path = os.path.dirname(os.path.dirname(dir_path))
                mask_path = os.path.join(base_path + "/ground_truth/")
                mask_path = os.path.join(mask_path, base_dir)
                mask_file_name = file_name.split(".")[0] + "_mask.png"
                mask_path = os.path.join(mask_path, mask_file_name)
                mask = self.load_mask(mask_path)

            return {"img": image, "mask": mask}

    def load_mask(self, mask_path):
        mask = Image.open(mask_path).convert("L")
        mask_transform = transforms.Compose([
            transforms.Resize((self.resize_shape[1], self.resize_shape[0])),
            transforms.ToTensor(),
        ])
        mask = mask_transform(mask)
        mask = (mask > 0).float()

        return mask
