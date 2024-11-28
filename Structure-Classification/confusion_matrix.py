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

# Scikit-learn imports
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             classification_report)

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Parameters:
    cm (numpy.ndarray): Confusion matrix to be plotted.
    classes (list): List of class labels.
    normalize (bool): Whether to normalize the confusion matrix.
    title (str): Title of the plot.
    cmap (colormap): Colormap to use for the plot.
    """
    plt.figure(figsize=(15,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


target_names = []

for key in train_generator.class_indices:
    target_names.append(key)

print(target_names)

Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(test_generator.classes, y_pred)

# Set the figure size to a smaller value (e.g., 5x5)
plt.figure(figsize=(7,7))

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

