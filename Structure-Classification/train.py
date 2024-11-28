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
# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
# Keras EfficientNet import (if needed, but be cautious with compatibility)
from keras_efficientnets import EfficientNetB0

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, 55 + 1)

plt.figure(figsize=(14, 5))

# Plot Training and Validation Accuracy with green and red colors
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, color='green', label='Training Accuracy')
plt.plot(epochs_range, val_acc, color='red', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot Training and Validation Loss with orange and blue colors
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, color='orange', label='Training Loss')
plt.plot(epochs_range, val_loss, color='blue', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


history = model.fit(   
    train_generator,
    epochs=55, 
    steps_per_epoch=train_generator.samples//BATCH_SIZE, 
    validation_data=validation_generator, 
    validation_steps=validation_generator.samples//BATCH_SIZE,  
    verbose=1,
)
