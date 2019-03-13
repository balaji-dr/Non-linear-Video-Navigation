from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import redis
import pickle


def image_to_feature_vector(image, size=(32, 32)):
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])
	
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	else:
		cv2.normalize(hist, hist)
	return hist.flatten()


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())


print("Describing images...")
imagePaths = list(paths.list_images(args["dataset"]))


rawImages = []
features = []
labels = []


for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

	if i > 0 and i % 1000 == 0:
		print("Processed {}/{}".format(i, len(imagePaths)))


rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)


print("Pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("Features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))


(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)


print("Evaluating raw pixel accuracy...")
model1 = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model1.fit(trainRI, trainRL)
acc = model1.score(testRI, testRL)
print("Raw pixel accuracy: {:.2f}%".format(acc * 100))


print("Evaluating histogram accuracy...")
model2 = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model2.fit(trainFeat, trainLabels)
acc = model2.score(testFeat, testLabels)
print("Histogram accuracy: {:.2f}%".format(acc * 100))


r = redis.Redis(host='localhost', port=6379, db=0)
pickled_object1 = pickle.dumps(model1)
model_stored1 = r.set('model1', pickled_object1)
pickled_object2 = pickle.dumps(model2)
model_stored2 = r.set('model2', pickled_object2)
print("The trained model is stored -> ", model_stored1,model_stored2)
