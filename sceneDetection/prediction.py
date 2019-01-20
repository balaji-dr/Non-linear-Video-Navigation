import redis
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


r = redis.Redis(host='localhost', port=6379, db=0)
model = unpacked_object = pickle.loads(r.get("model"))


image = cv2.imread("/Users/balajidr/Developer/FYP_TEMP/sceneDetection/trainingdata/slide.627.jpg")
hist = extract_color_histogram(image)
prediction = model.predict([hist])
print("Predicted class is -> ", str(prediction[0]).upper())
