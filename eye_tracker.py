import cv2
from main import logging
import numpy as np
import time
import pyautogui

# Define the Haar Cascade paths for face and eye detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_mcs_mouth.xml")

# Define the last known pupil position
last_known_pupil = {}

"""------------------------ Eye Tracking ------------------------
The function first converts the frame to grayscale and then detects faces
in the frame. For each face, the function detects eyes and draws bounding
boxes around them. The function also calls the `detect_pupil` function to
detect the pupil in each eye.
"""
def track_eyes(frame):
    logging.info("Tracking eyes...")
    cv2.imshow("Eye Tracking", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_color, scaleFactor=1.07, minNeighbors=45, minSize=(20, 20))
        for idx, (ex, ey, ew, eh) in enumerate(eyes):
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(
                roi_color, "eye",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            modified_eye_region = detect_pupil(eye_roi, idx)
            detect_mouth(roi_color)
            roi_color[ey:ey+eh, ex:ex+ew] = modified_eye_region

    return frame


"""------------------------ Pupil Detection ------------------------
The function first converts the eye region to grayscale and then applies
adaptive thresholding to the eye region. It then uses the HoughCircles
function to detect circles in the eye region. If circles are detected, the
function draws a bounding box around the circle and marks the center of the
circle. If no circles are detected, the function uses the last known pupil
position to draw a circle and mark the center of the pupil.
"""
def detect_pupil(eye_region, eye_index):
    
    # Retrieve the values from the trackbars
    param1 = cv2.getTrackbarPos('Param1', 'Parameters')+1
    param2 = cv2.getTrackbarPos('Param2', 'Parameters')+1

    global last_known_pupil
    logging.info("Detecting pupil...")

    eye_region_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    thresh_eye = cv2.adaptiveThreshold(
        eye_region_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    if param1 == -1 or 0 and param2 == -1 or 0:
        param1, param2 = 100, 30
    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        thresh_eye,
        cv2.HOUGH_GRADIENT, 1, 20,
        param1=param1,
        param2=param2,
        minRadius=5,
        maxRadius=30)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        logging.info(f"Detected {circles.shape[1]} circles.")
        i = circles[0, 0]
        last_known_pupil[eye_index] = (i[0], i[1])
        cv2.circle(eye_region, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(eye_region, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        logging.info("No circles detected.")
        i = last_known_pupil.get(eye_index, (0, 0))
        # Using 10 as a default radius
        cv2.circle(eye_region, (i[0], i[1]), 10, (0, 255, 0), 2)
        cv2.circle(eye_region, (i[0], i[1]), 2, (0, 0, 255), 3)
    return eye_region


"""------------------------ Mouth Detection ------------------------
The function first converts the frame to grayscale and then detects faces
in the frame. For each face, the function detects mouths and draws bounding"""
def detect_mouth(frame):
    logging.info("Detecting mouth...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouths = mouth_cascade.detectMultiScale(
        gray, scaleFactor=1.07, minNeighbors=45, minSize=(80, 100))
    for (x, y, w, h) in mouths:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
                frame, "smile",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

"""------------------------ Trackbar------------------------
This function is used as a placeholder for the trackbar functions."""""
def nothing(x):
    pass

cv2.namedWindow("Parameters", cv2.WINDOW_NORMAL)
cv2.createTrackbar('Param1', 'Parameters', 100, 300, nothing)
cv2.createTrackbar('Param2', 'Parameters', 30, 100, nothing)



"""------------------------ Run ------------------------
This is the main function of the program. It opens the default camera and
starts the calibration process. It then starts the webcam feed and tracks
the eyes in the feed."""
def run():
    global param1, param2
    # Open default camera
    cap = cv2.VideoCapture(1)
    video_feed_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_feed_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logging.info("Video feed dimensions: "
                 f"{video_feed_width}x{video_feed_height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = track_eyes(frame)
        cv2.imshow("Eye Tracking", frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


run()
