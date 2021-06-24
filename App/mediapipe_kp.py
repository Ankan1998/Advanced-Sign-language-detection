import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe as mp 

# Mediapipe Holistic Introduction
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

#prediction Function using mediapipe
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results

#Draw on prediction
def draw_on_prediction(image,results):
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

# Access webcam using opencv
def main_func():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
        while cap.isOpened():

            #read feed
            ret, frame = cap.read()

            # Make prediction
            image, results = mediapipe_detection(frame, holistic)

            # Draw Connections
            draw_on_prediction(image,results)

            #print(results)
            cv2.imshow("'Holistic feed", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()




if __name__ == "__main__":
    main_func()
