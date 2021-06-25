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

# Draw landmark with formatting
def draw_formatted_landmark(image,results):
    mp_drawing.draw_landmarks(
    image, 
    results.face_landmarks, 
    mp_holistic.FACE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(80,120,50), thickness =1, circle_radius=1),
    mp_drawing.DrawingSpec(color=(90,200,111), thickness =1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(
    image, 
    results.left_hand_landmarks, 
    mp_holistic.HAND_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(40,10,210), thickness =2, circle_radius=2),
    mp_drawing.DrawingSpec(color=(180,96,72), thickness =2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
    image, 
    results.right_hand_landmarks, 
    mp_holistic.HAND_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(40,10,210), thickness =2, circle_radius=2),
    mp_drawing.DrawingSpec(color=(180,96,72), thickness =2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(
    image, 
    results.pose_landmarks, 
    mp_holistic.POSE_CONNECTIONS,
    mp_drawing.DrawingSpec(color=(130,140,10), thickness =2, circle_radius=2),
    mp_drawing.DrawingSpec(color=(80,36,14), thickness =2, circle_radius=2)
    )

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
            # draw_on_prediction(image,results)

            # Draw formatted Landmark
            draw_formatted_landmark(image,results)

            #print(results)
            cv2.imshow("'Holistic feed", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    return results


def keypoint_ext(results, landmark="pose"):    
    fin_arr = np.array([])
    if landmark.lower() == "pose":
        for res in results.pose_landmarks.landmark:
            fin_arr = np.append(fin_arr, [res.x,res.y,res.z]).flatten()

        return fin_arr

    if landmark.lower() == "face":
        for res in results.face_landmarks.landmark:
            fin_arr = np.append(fin_arr, [res.x,res.y,res.z]).flatten()

        return fin_arr




if __name__ == "__main__":
    res = main_func()
    arr_pose = keypoint_ext(res, "pose")
    arr_face = keypoint_ext(res, "face")

