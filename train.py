import cv2
import os
import numpy as np
import logging
import json

# Initialize face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Directory containing the dataset
dataset_path = 'dataset/'

# Lists to hold faces and labels
faces = []
labels = []
label_counter = 0

# Mapping of label numbers to person names, based on folder names
label_map = {}


# Iterate through dataset directories and images
def run():
    global label_counter
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)

        if os.path.isdir(person_path):
            # Assign the current label for the person
            label = label_counter
            # Map the label number to the person name
            label_map[label] = person_name
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                # Convert to grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Detect faces in the image
                detected_faces = face_cascade.detectMultiScale(
                    image, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in detected_faces:
                    face = image[y:y+h, x:x+w]
                    faces.append(face)
                    labels.append(label)
            # Increment the label counter after processing folder
            label_counter += 1

    logging.info(f"Found {len(faces)} faces from {len(set(labels))} persons.")
    # Initialize the LBPH face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the recognizer
    face_recognizer.train(faces, np.array(labels))

    # Save the trained model
    model_path = "face_recognizer_model.yml"
    face_recognizer.save(model_path)

    # write out label_map to json file
    # Convert integer keys to string because json does not support integer keys
    str_label_map = {str(key): value for key, value in label_map.items()}
    with open('label_map.json', 'w') as label_map_json:
        json.dump(str_label_map, label_map_json)
    logging.info(f"Saved label map: {str_label_map}")
