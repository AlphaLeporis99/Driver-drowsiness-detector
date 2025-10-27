import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame
import time
# Initialize the camera and take the instance
cap = cv2.VideoCapture(0)
# Initialize the face detector and landmark detector
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("C:/Users/abhay/Driver-Drowsiness-Detection-System/shape_predictor_5_face_landmarks.dat")
# Status marking for the current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
# Load the warning sound file using pygame
pygame.mixer.init()
warning_sound = pygame.mixer.Sound("C:/Users/abhay/Driver-Drowsiness-Detection-System/warning_sound.wav")
# Function to calculate distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist
# Function to detect eye blinking
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0
face_frame=None

# Initialize timer variables
start_time = time.time()
current_time = 0
# Define the text coordinates (x, y) for displaying the "take a break" message
x = 100
y = 100
# Define the message coordinates
message_x = 10
message_y = 70
# Define the message
message = ""
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray, dtype=np.uint8)        # to fix image format issues with dlib library
    faces = detector(gray)
    # Calculate current time
    current_time = int(time.time() - start_time)
    # Format the time as "mm:ss"
    timer_text = f"{current_time//60:02d}:{current_time%60:02d}"
    for face in faces:
        # Extracting face coordinates
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Check if it's been 1 minute
        if current_time == 10:
            message = "You have been driving for 10 seconds. You should take a break!"
            color = (0, 0, 255)
            # Play the warning sound
        # Predicting facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        # Detecting blinks for left and right eyes
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        # Determine the status based on the eye blinks
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                # Play the warning sound
                warning_sound.play()
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
        # Display the status text and draw facial landmarks
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, "Time: " + timer_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adjust the font scale (0.5)
        cv2.putText(frame, message, (message_x,message_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Display the frames with detected face and facial landmarks
    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press the ESC (Escape) Key
        break
# Release the camera and stop the warning sound
cap.release()
cv2.destroyAllWindows()
