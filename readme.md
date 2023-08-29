# Face Recognition with Python and OpenCV

This project provides a simple face recognition system using Python and OpenCV. The system can train on a dataset of images, save the trained model, and then use the model to recognize faces in real-time from a webcam feed.

## Project Structure

- `main.py`: The main script that provides an interactive menu to train the recognizer or start webcam face recognition.
- `train.py`: The script responsible for training the face recognizer using a dataset of images stored in a specific directory structure.
- `webcam_recognition.py`: The script that accesses the webcam feed, detects faces, and recognizes them using the trained model.

## Features

- **Face Detection**: Detects faces in images using Haar cascades.
- **Face Recognition**: Uses the Local Binary Patterns Histogram (LBPH) face recognizer to recognize faces.
- **Interactive Menu**: An interactive menu in the main script to guide the user through training and recognition processes.
- **Logging**: Logs important events and data for debugging and analysis.

## How to Use

1. **Setup**:
    - Clone the repository to your local machine.
    - Ensure you have OpenCV installed with the `face` module.
    - Create a directory named `dataset` in the project root.

2. **Prepare the Dataset**:
    - Inside the `dataset` directory, create sub-directories for each person you want to recognize. The name of the sub-directory will be used as the label for the person.
    - Place several clear images of each person's face inside their respective sub-directory. Make sure the images only contain the face of the person you're trying to recognize.

3. **Training**:
    - Run `main.py`.
    - Choose the option to train the face recognizer. The script will process the images, detect faces, and train the recognizer.
    - Once training is complete, the model will be saved as `face_recognizer_model.yml`.

4. **Recognition**:
    - Run `main.py` again.
    - Choose the option to start webcam face recognition.
    - The script will access your webcam, detect faces in real-time, and try to recognize them based on the trained model.
    - Press 'q' to exit the webcam feed.

## Notes

- Ensure proper lighting when capturing images for the dataset and during the recognition process.
- The accuracy of recognition may vary based on the quality and variety of training images, as well as real-time conditions during recognition.
- You can easily extend this project to use other face recognition algorithms provided by OpenCV or even integrate deep learning models.

## Dependencies

- Python
- OpenCV (with `face` module)

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute as per the license terms.
