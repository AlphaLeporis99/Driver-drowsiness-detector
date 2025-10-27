# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 17:12:03 2025

@author: abhay
"""

import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize face detector and 68-point landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/abhay/Driver-Drowsiness-Detection-System/shape_predictor_68_face_landmarks.dat")

# Status variables
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Load warning sound
pygame.mixer.init()
warning_sound = pygame.mixer.Sound("C:/Users/abhay/Driver-Drowsiness-Detection-System/warning_sound.wav")

# Functions
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0

# Timer
start_time = time.time()

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    faces = detector(gray)

    current_time = int(time.time() - start_time)
    timer_text = f"{current_time//60:02d}:{current_time%60:02d}"

    face_frame = frame.copy()  # Default in case no face

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 10-second warning
        message = ""
        if current_time == 10:
            message = "You have been driving for 10 seconds. Take a break!"
            color = (0, 0, 255)
            warning_sound.play()

        # Landmarks
        shape = predictor(gray, face)
        landmarks = face_utils.shape_to_np(shape)

        if len(landmarks) != 68:
            print(f"Warning: Detected {len(landmarks)} landmarks, expected 68")
            continue

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Status logic
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                warning_sound.play()
        elif left_blink == 1 or right_blink == 1:
            sleep = active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
        else:
            drowsy = sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display info
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, "Time: " + timer_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Face Detection", face_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
