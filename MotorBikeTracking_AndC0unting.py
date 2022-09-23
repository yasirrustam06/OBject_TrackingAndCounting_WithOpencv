import cv2
import numpy as np
from tracker import *



cap = cv2.VideoCapture('highway.mp4')

# This Is Actually backGround Subtractor Alghorithm we are using Here As a Object Tracker....

object_tracker = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

# Create tracker object....
tracker = EuclideanDistTracker()



while True:
    ret, frame = cap.read()
    # print(frame.shape)
    # Take the Region Of Interests  Area Out of Our Frame Window....

    roi = frame[340: 720,500: 800]

    # Apply object Tracker to Subtract the background....
    mask = object_tracker.apply(roi)

    # Apply threshold to take the white object from window....
    _,thresh = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)

    # Find The Contour Of our mask Image Window....
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #  Drawing The Contour....
    Detection_Coordinates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)

        if area > 100:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            Detection_Coordinates.append([x,y,w,h])
            # print(Detection_Coordinates)


    boxes_ids = tracker.update(Detection_Coordinates)
    for box_id in boxes_ids:
        # print(box_id)
        X_Id, Y_Id, W_Id, H_Id, ID = box_id

        # Drawing the rectular box and counting the bikes
        cv2.putText(roi, str(ID), (X_Id,Y_Id - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi,(X_Id,Y_Id),(X_Id+W_Id,Y_Id+H_Id), (0, 255, 0), 3)




    cv2.imshow("Roi",roi)
    cv2.imshow("Mask",thresh)
    cv2.imshow("Image",frame)
    cv2.waitKey(1)














