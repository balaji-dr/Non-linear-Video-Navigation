import cv2
import os
from progress.bar import Bar

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR)


def extract_all_frames(video_path: str) -> str:
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        print(" ---> Video has been successfully detected.")
    else:
        print(" ---> Cannot find video in the given path.")
    count = 0
    i = 1
    if not os.path.exists(BASE_DIR+"/SlideExtraction/frames"):
        os.makedirs(BASE_DIR+"/SlideExtraction/frames")
    print(" --> Created directory to store all the frames.")
    print(" --->  Extracting all the frames from the video.")
    extraction_bar = Bar('Extracting', max=25000)
    while success:
        if count % 30 == 0:
            timestamp = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            cv2.imwrite(BASE_DIR + '/SlideExtraction/frames/' + 'frame%d.jpg' % i, image)
            # print('Extracted Frame no %d of Time %d ' % (count, timestamp))
            i += 1
            extraction_bar.next()
        success, image = vidcap.read()
        count += 1
    extraction_bar.finish()
    return BASE_DIR+'/SlideExtraction/frames'
