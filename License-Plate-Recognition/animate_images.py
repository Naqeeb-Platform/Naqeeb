import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # providing a progress bar to visualize the iteration progress
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image

images1=[]
for i in tqdm(range(len(image_paths))):#
    image_path=image_paths[i]
    image=cv2.imread(image_path)
    images1+=[image]

from matplotlib import animation, rc
rc('animation', html='jshtml')

def create_animation(ims):
    fig=plt.figure(figsize=(10,6))
    #plt.axis('off')
    im=plt.imshow(cv2.cvtColor(ims[0],cv2.COLOR_BGR2RGB))
    plt.close()
    def animate_func(i):
        im.set_array(cv2.cvtColor(ims[i],cv2.COLOR_BGR2RGB))
        return [im]
    return animation.FuncAnimation(fig, animate_func, frames=len(ims), interval=1000//4)

create_animation(images1)
