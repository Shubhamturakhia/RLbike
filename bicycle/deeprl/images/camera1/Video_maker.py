import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('C:/Users/Shubh/OneDrive/Desktop/THESIS + MORE/Thesis/bicycle/deeprl/images/camera1/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('RLbike_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()