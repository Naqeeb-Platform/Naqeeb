import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # providing a progress bar to visualize the iteration progress
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image
import ultralytics
from ultralytics import YOLO


image_paths=[] # store tha lisence plate images path
text_paths=[] # store the bounding boxes associated with each image
for dirname, _, filenames in os.walk('/kaggle/input/saudi-license-plate-characters/License-Characters-by-2-27classes/train'):
    for filename in filenames:
        if filename[-5:]=='.jpeg' or filename[-4:]=='.png':  # image
            image_paths+=[os.path.join(dirname, filename)]
        elif filename[-4:]=='.txt': # txt
            text_paths+=[os.path.join(dirname, filename)]

for dirname, _, filenames in os.walk('/kaggle/input/saudi-license-plate-characters/License-Characters-by-2-27classes/test'):
    for filename in filenames:
        if filename[-5:]=='.jpeg' or filename[-4:]=='.png':  # image
            image_paths+=[os.path.join(dirname, filename)]
        elif filename[-4:]=='.txt': # txt
            text_paths+=[os.path.join(dirname, filename)]
image_paths=sorted(image_paths)
text_paths=sorted(text_paths)
print(image_paths[0])
print(text_paths[0])
print(len(image_paths))
print(len(text_paths))
