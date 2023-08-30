import cv2
import logging
import numpy as np
import time

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

last_known_pupil = {}

"""
The function first converts the frame to grayscale and then detects faces
in the frame. For each face, the function detects eyes and draws bounding
boxes around them. The function also calls the `detect_pupil` function to
detect the pupil in each eye.
"""
def track_eyes(frame):
    logging.info("Tracking eyes...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.07, minNeighbors=45, minSize=(20, 20))
        for idx, (ex, ey, ew, eh) in enumerate(eyes):
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
            modified_eye_region = detect_pupil(eye_roi, idx)
            roi_color[ey:ey+eh, ex:ex+ew] = modified_eye_region

    return frame


"""
The function first converts the eye region to grayscale and then applies
adaptive thresholding to the eye region. It then uses the HoughCircles
function to detect circles in the eye region. If circles are detected, the
function draws a bounding box around the circle and marks the center of the
circle. If no circles are detected, the function uses the last known pupil
position to draw a bounding box and mark the center of the pupil.
"""
def detect_pupil(eye_region, eye_index):
    global last_known_pupil
    logging.info("Detecting pupil...")

    eye_region_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    thresh_eye = cv2.adaptiveThreshold(
        eye_region_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        thresh_eye,
        cv2.HOUGH_GRADIENT, 1, 20,
        param1=100,
        param2=30,
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
        logging.warning("No circles detected.")
        i = last_known_pupil.get(eye_index, (0, 0))
        # Using 10 as a default radius
        cv2.circle(eye_region, (i[0], i[1]), 10, (0, 255, 0), 2)  
        cv2.circle(eye_region, (i[0], i[1]), 2, (0, 0, 255), 3)

    return eye_region


"""
This part of the code is building a gaze-based interaction system. The
calibration process is used to map the pupil position to the screen
coordinates. The calibration process is done by displaying a marker on the
screen and asking the user to focus on the marker. The pupil position is
then recorded and mapped to the screen coordinates. This process is repeated
for a few points on the screen. The calibration points are the center of the
screen, top-left, top-right, bottom-left, and bottom-right.
"""

# Adjust for your screen resolution
screen_width = 1719
screen_height = 1112

calibration_points = 5

calibration_coords = [
    (screen_width // 2, screen_height // 2),  # center
    (0, 0),                                   # top-left
    (screen_width, 0),                        # top-right
    (0, screen_height),                       # bottom-left
    (screen_width, screen_height)             # bottom-right
]
pupil_positions = []


"""
This function runs the calibration process. It displays a marker on the
screen and asks the user to focus on the marker. The pupil position is then
recorded and mapped to the screen coordinates. This process is repeated for
a few points on the screen. The calibration points are the center of the
screen, top-left, top-right, bottom-left, and bottom-right.
"""
def calibration(cap):
    logging.info("Starting calibration...")
    for coord in calibration_coords:
        logging.info(f"Calibrating at {coord}")
        display_marker(*coord)
        # Give the user time to focus on the marker
        time.sleep(2)  

        # Capture a frame and get the pupil position
        ret, frame = cap.read()
        logging.info("Capturing frame...")
        if not ret:
            logging.error("Failed to capture frame.")
            break
        frame = track_eyes(frame)
        # Assuming you get the pupil position from the `track_eyes` function
        logging.info(f"Getting pupil position: {last_known_pupil.get(0, (0, 0))}")
        pupil_positions.append(last_known_pupil.get(0, (0, 0)))

    # Close the calibration window
    logging.info("Closing calibration window...")
    cv2.destroyAllWindows()


"""
This function displays a marker on the screen at the given coordinates. The
marker is a green dot with a radius of 50 pixels.
"""
def display_marker(x, y):
    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", SCREEN_WIDTH, SCREEN_HEIGHT)
    cv2.moveWindow("Calibration", 0, 0)
    blank_image = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    # Green dot as the marker
    cv2.circle(blank_image, (x, y), 10, (0, 255, 0), -1)  
    cv2.imshow("Calibration", blank_image)
    # Wait for 100 ms to update window content
    cv2.waitKey(100)  


"""
This is the main function of the program. It opens the default camera and
starts the calibration process. It then starts the webcam feed and tracks
the eyes in the feed."""
def run():
    # Open default camera
    cap = cv2.VideoCapture(1)  

    # Start the calibration process
    calibration(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = track_eyes(frame)
        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


run()
