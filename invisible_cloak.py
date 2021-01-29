import cv2
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
# Input argument
parser.add_argument(
    "--video", help="Path to input video file. Skip this argument to capture frames from a camera.")

args = parser.parse_args()
cap = cv2.VideoCapture(args.video if args.video else 0)

time.sleep(3)
count = 0
background = 0

for i in range(30):
    ret, background = cap.read()


while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    count += 1

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([110, 255, 255])
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

    mask1 = mask1+mask2

    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8), iterations=2)
    mask1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    mask2 = cv2.bitwise_not(mask1)

    res1 = cv2.bitwise_and(background, background, mask=mask1)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow('Magic !!!', final_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
