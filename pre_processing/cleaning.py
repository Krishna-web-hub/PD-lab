import os
import cv2

dataset = r"D:\...\Images"

for root, dirs, files in os.walk(dataset):
    for f in files:
        path = os.path.join(root,f)
        img = cv2.imread(path)

        if img is None:
            print("Corrupted:", path)