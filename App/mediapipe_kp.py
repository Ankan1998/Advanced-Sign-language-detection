import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp 

# Mediapipe Holistic Introduction
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writiable = False
    results = model.process(image)
    image.flags.writable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results

# Access webcam using opencv
cap = cv2.VideoCapture(0)
while cap.isOpened():
    #read feed
    ret, frame = cap.read()
    cv2.imshow('OpenCv Feed', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
