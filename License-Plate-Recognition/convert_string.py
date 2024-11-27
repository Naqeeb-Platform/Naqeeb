import os
import numpy as np
import pandas as pd
from shutil import copyfile

N=list(range(27))   
Names2 = ['0', '1', '2', '3', '4', '5', '6', '7','8', '9', 'A', 'B', 'D', 'E', 'G', 'H', 'J', 'K', 'L', 'N', 'R', 'S', 'T', 'U', 'V', 'X', 'Z']
class_map=dict(zip(N,Names2)) # integer from 0 to 26 mapped to string  from 00 to 26
print(class_map)

# replaces the first col from int to string (from previous class_map)
for i in range(len(BOX)):
    BOX.iloc[i,0]=class_map[int(BOX.iloc[i,0])]
display(BOX)
display(BOX.iloc[:,0].value_counts())
