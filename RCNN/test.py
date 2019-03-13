import os
import cv2
import pandas as pd
print("Drawing boundary on image")
# open image file
xmin=4
ymin=33
xmax=184
ymax=170
img = cv2.imread("/media/karthi/Empty/Non-linear-Video-Navigation/RCNN/chart1.png")
cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(255,0,0),15)
cv2.imwrite("/media/karthi/Empty/Non-linear-Video-Navigation/RCNN/testing.jpg",img)
