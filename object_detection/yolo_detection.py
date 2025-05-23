
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image_path):
    results = model(image_path)
    results.print()
    results.show()
    return results
