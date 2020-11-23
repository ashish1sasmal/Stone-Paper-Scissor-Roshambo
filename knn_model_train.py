import time
s1 = time.time()

from KNNModel.knn import KNN
import cv2
import numpy as np
import sys

model = KNN(2, -1, "Images", 32, "stone-paper-scissor")

model.classify()