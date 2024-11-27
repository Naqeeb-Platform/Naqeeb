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


boxdata=[] # information of each bbox
boxfile=[] # file names (without the full path)
for i in range(len(text_paths)):
    file=text_paths[i]
    boxdata+=[np.loadtxt(file)]
    boxfile+=[file[0:-4].split('/')[-1]]
print(boxdata[0])
print(boxfile[0])


BOX=pd.DataFrame()
for i in range(len(boxdata)):
    if type(boxdata[i][0])==np.float64: # if its a 1d list
        add=pd.DataFrame([boxdata[i]])
        add[5]=boxfile[i]
        BOX=pd.concat([BOX,add])
    else: # if its 2d with multiple bboxes
        add=pd.DataFrame(boxdata[i])
        add[5]=boxfile[i]
        BOX=pd.concat([BOX,add])
BOX=BOX.reset_index(drop=True) # reset index and start from 0
display(BOX)
