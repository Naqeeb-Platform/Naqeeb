import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # providing a progress bar to visualize the iteration progress
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image

def text_pathing(text_path):

    with open(text_path, 'r') as file:
        text_data = file.read()
    
    newText=text_path.split('/')[-1]
    path=f'./processed images/{newText}'
    with open(path, 'w') as file:
        file.write(text_data)
    return path

new_text_path=[]
for i in text_paths :
    new_text_path+=[text_pathing(i)]

new_text_path=sorted(new_text_path)
new_images_path=sorted(new_images_path)

print(new_text_path[0])
print(new_images_path[0])


print(len(new_images_path))
print(len(new_text_path))

