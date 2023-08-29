import cv2
import logging
import json

# Load the trained recognizer and cascade classifier for face detection
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer_model.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read label_map from json file
with open('label_map.json', 'r') as fp:
    str_label_map = json.load(fp)
    logging.info(f"Loaded label map: {str_label_map}")

# Convert string keys back to integers
label_map = {int(k): v for k, v in str_label_map.items()}


def run():
    # Access the webcam feed
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        logging.error("Could not open video capture device.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_roi)
            print(f"Label: {label}, Confidence: {confidence}")
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_map[label]} - {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
