import os
import sys
import glob
import random
import shutil
import pathlib
import requests
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from struct import unpack
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance
from IPython.display import FileLink

def split_dataset(src_dir, dest_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Splits images in src_dir into train, val, and test sets."""
    # Ensure the destination structure for each class in train, val, and test
    for class_folder in os.listdir(src_dir):
        class_path = os.path.join(src_dir, class_folder)
        if os.path.isdir(class_path):  # Only process directories (classes)
            # List all images in the class directory
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

            # Shuffle images
            random.shuffle(images)

            # Calculate split sizes
            total_images = len(images)
            train_count = int(total_images * train_ratio)
            val_count = int(total_images * val_ratio)
            test_count = (total_images - train_count - val_count)  # Remainder goes to test set

            # Split the images
            train_images = images[:train_count]
            val_images = images[train_count:train_count + val_count]
            test_images = images[train_count + val_count:]

            # Process each split and create directories as needed
            for split_name, split_images in zip(["train", "val", "test"], [train_images, val_images, test_images]):
                split_class_dir = os.path.join(dest_dir, split_name, class_folder)

                # Create the split/class folder if it doesn't exist
                os.makedirs(split_class_dir, exist_ok=True)

                # Copy files to the respective directories
                for img in split_images:
                    src_img_path = os.path.join(class_path, img)
                    dest_img_path = os.path.join(split_class_dir, img)
                    try:
                        shutil.copy(src_img_path, dest_img_path)
                    except FileNotFoundError as e:
                        print(f"File not found error: {e}")
                        print(f"Source image path: {src_img_path}")
                        print(f"Destination image path: {dest_img_path}")
                    except Exception as e:
                        print(f"Error copying file '{img}': {e}")

            print(f"Class '{class_folder}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images.")

src_directory = "/kaggle/input/classification-dataset-final/classification_dataset_final"
dest_directory = "/kaggle/working/classification_dataset_final33"


split_dataset(src_directory, dest_directory)
print("Dataset split into train, val, and test sets.")
