import ultralytics
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

source = '/kaggle/input/unseen-data'

# Run inference on the source
results = model(source, save_txt = True)
