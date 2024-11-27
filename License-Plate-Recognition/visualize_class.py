import numpy as np
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt
from PIL import Image


def visualization_frequency_of_classes(class_frequency):
    # Plotting
    plt.figure(figsize=(10, 6))
    class_frequency.plot(kind='bar', color='skyblue')
    plt.title('Frequency of Each Class')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  
    plt.grid(axis='y', linestyle='--', alpha=0.7) 
    plt.tight_layout()
    figure_path = os.path.join('/kaggle/working/', "Frequency of Each Class'.png")  # Adjust the file name and extension as needed
    plt.savefig(figure_path)
    plt.show()

class_frequency = BOX.iloc[:, 0].value_counts()
visualization_frequency_of_classes(class_frequency)
