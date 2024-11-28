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

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

CLASSES = 6
BATCH_SIZE = 16
IMAGE_SIZE = [224, 224]

train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=120,         # Cars underneath can be slightly rotated
    width_shift_range=0.2,     # Car might not be perfectly centered
    height_shift_range=0.2,    # Same for vertical positioning
    zoom_range=0.15,          # Different camera distances
    brightness_range=[0.7,1.3], # Important for under-vehicle lighting variations
    shear_range=0.1,          # Slight perspective changes
    fill_mode='constant',     
   
)




train_generator = train.flow_from_directory(
    "/kaggle/input/classificationdataset/train", 
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='categorical',
    subset="training"
)
test_val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_val_datagen.flow_from_directory(
    "/kaggle/input/classificationdataset/val",
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Create the test data generator without augmentations
test_generator = test_val_datagen.flow_from_directory(
    "/kaggle/input/classificationdataset/test",
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)


