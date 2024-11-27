import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # providing a progress bar to visualize the iteration progress
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image

import random

data = list(zip(new_images_path, new_text_path))
random.shuffle(data)

total_images = len(data)
train = int(0.70 * total_images)
test = int(0.20 * total_images)
valid = total_images - train - test


for i, (image_path, text_path) in enumerate(data):
    image_file = image_path.split('/')[-1]
    text_file = text_path.split('/')[-1]

    if i < train:
        copyfile(image_path, f'datasets/train/{image_file}')
        copyfile(text_path, f'datasets/train/{text_file}')
    elif train <= i < (train + valid):
        copyfile(image_path, f'datasets/valid/{image_file}')
        copyfile(text_path, f'datasets/valid/{text_file}')
    else:
        copyfile(image_path, f'datasets/test/{image_file}')
        copyfile(text_path, f'datasets/test/{text_file}')

print(f"Total images: {total_images}")
print(f"Train images: {train}")
print(f"Validation images: {valid}")
print(f"Test images: {test}")
