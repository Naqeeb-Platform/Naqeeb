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

def count_images(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):  
            num_images = len(os.listdir(class_dir))  
            class_counts[class_name] = num_images
    return class_counts

base_path = '/kaggle/input/wedyannnnnnnnnn/full_classification2'
train_path = os.path.join(base_path, 'train')
val_path = os.path.join(base_path, 'val')
test_path = os.path.join(base_path, 'test')



train_counts = {}
val_counts = {}
test_counts = {}



train_counts = count_images(train_path)
val_counts = count_images(val_path)
test_counts = count_images(test_path)



total_counts = {}
for class_name in set(train_counts.keys()).union(val_counts.keys(), test_counts.keys()):
    total_counts[class_name] = {
        'train': train_counts.get(class_name, 0),
        'val': val_counts.get(class_name, 0),
        'test': test_counts.get(class_name, 0),
        'total': train_counts.get(class_name, 0) + val_counts.get(class_name, 0) + test_counts.get(class_name, 0)

    }



total_train = sum(total['train'] for total in total_counts.values())
total_val = sum(total['val'] for total in total_counts.values())
total_test = sum(total['test'] for total in total_counts.values())
total_dataset = total_train + total_val + total_test



print(f"\nTotal Dataset Size: {total_dataset}")
print(f"Total Training Images: {total_train}")
print(f"Total Validation Images: {total_val}")
print(f"Total Testing Images: {total_test}")
