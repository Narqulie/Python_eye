import logging

logging.basicConfig(filename="app.log",
                    level="DEBUG",
                    filemode="w",
                    format=(
                        "%(asctime)s"
                        " - %(name)s"
                        " - %(levelname)s"
                        " - %(message)s")
                    )

"""
Main entry point of the program. This is where the user is presented with
options to either train the face recognizer or start the webcam face
recognition. The user can also exit the program.
"""


def main():
    while True:
        logging.info("Starting face recognition program.")
        print("Choose an option:")
        print("1. Train face recognizer")
        print("2. Start webcam face recognition")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            logging.info("Training face recognizer...")
            # Import training module here to avoid circular imports
            import train
            # Run the training script
            train.run()

        elif choice == "2":
            logging.info("Starting webcam face recognition...")
            # Import webcam_recognition module here to avoid circular imports
            import webcam_recognition
            # Run the webcam face recognition script
            webcam_recognition.run()
        elif choice == "3":
            logging.info("Exiting...")
            # Exit the program
            break
        else:
            logging.warning("Invalid choice. Please try again.")
    logging.info("Program exited.")


# This is the standard boilerplate that calls the main() function.
if __name__ == "__main__":
    main()
