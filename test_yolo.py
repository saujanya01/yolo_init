import cv2
import numpy as np
import matplotlib.pyplot as plt

net = cv2.dnn.readNetFromDarknet("yolov3_ts_test.cfg",r"yolov3_ts_train_800.weights")

print(net)