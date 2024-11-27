import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # providing a progress bar to visualize the iteration progress
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image

def process_image(imaging):
    image_path = image_paths[imaging]
    image = cv2.imread(image_path)
    H, W = image.shape[0], image.shape[1]
    file = image_path[0:-4].split('/')[-1]
    
    if BOX[BOX[5] == file] is not None:
        boxes = BOX[BOX[5] == file]
        boxes = boxes.reset_index(drop=True)
        
    # convert to grayscale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Cubic interpolation
    image_interp = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Sharpening the image
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1], 
                                  [-1, -1, -1]])
    image_sharpened = cv2.filter2D(image_interp, -1, kernel_sharpening)
    
    
    pic=image_path.split('/')[-1]
    path=f'./processed images/{pic}'
    cv2.imwrite(filename=path,img=image_sharpened)
    
    return path ,image_sharpened
