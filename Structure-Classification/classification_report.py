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


# Scikit-learn imports
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             classification_report)

total_predictions = np.sum(cm)
correct_predictions = np.trace(cm)  # Sum of diagonal elements
incorrect_predictions = total_predictions - correct_predictions

print(f"Number of Incorrect Predictions: {incorrect_predictions}")

report = classification_report(y_true, y_pred, target_names=target_names, labels=np.arange(len(target_names)))
print(report)
