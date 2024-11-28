

label_image_path = "/kaggle/working/runs/detect/train32/val_batch0_labels.jpg"
pred_image_path = "/kaggle/working/runs/detect/train32/val_batch0_pred.jpg"

# Load images
label_image = mpimg.imread(label_image_path)
pred_image = mpimg.imread(pred_image_path)
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm # providing a progress bar to visualize the iteration progress
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image

# Plot images
plt.figure(figsize=(8, 12))  # Adjust the figure size as needed

# Plot label image
plt.subplot(2, 1, 1)
plt.imshow(label_image)
plt.title('True Labels')
plt.axis('off')

# Plot predicted image
plt.subplot(2, 1, 2)
plt.imshow(pred_image)
plt.title('Predicted Labels')
plt.axis('off')

plt.tight_layout() 
plt.show()

confusion_matrix = "/kaggle/working/runs/detect/train32/confusion_matrix_normalized.png"
image = Image.open(confusion_matrix)
image=np.array(image)
plt.figure(figsize=(20,10))
plt.imshow(image)
plt.show()

