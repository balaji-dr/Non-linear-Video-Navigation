import redis
import os
import cv2
from imutils import paths
import argparse
import pickle
import imutils

def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()

import cv2
import pickle
import imutils


def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["dataset"]))

r = redis.Redis(host='localhost', port=6379, db=0)
model1 = unpacked_object = pickle.loads(r.get("model1"))
model2 = unpacked_object = pickle.loads(r.get("model2"))


for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    pixels = image_to_feature_vector(image)
    prediction = model1.predict([pixels])
    print("Predicted class is -> ", str(prediction[0]).upper())

print("Histogram")

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    hist = extract_color_histogram(image)
    prediction = model2.predict([hist])
    print("Predicted class is -> ", str(prediction[0]).upper())
    
