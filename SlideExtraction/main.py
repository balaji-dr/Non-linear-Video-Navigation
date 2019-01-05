import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
vidcap = cv2.VideoCapture('/Users/balajidr/Developer/FYP_TEMP/videos/trainvideo.mp4')
success, image = vidcap.read()
count = 0
while success:
    if count % 30 == 0:
        timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        cv2.imwrite(BASE_DIR + '/SlideExtraction/slides/' + 'frame%d - %d.jpg' % (count, timestamp), image)
        print('Extracted Frame %d of Time %d ' % (count, timestamp), success)
    success, image = vidcap.read()
    count += 1
