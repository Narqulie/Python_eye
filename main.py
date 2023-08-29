import logging

logging.basicConfig(filename="app.log",
                    level="INFO",
                    filemode="w",
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


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
            import train
            train.run()  # Run the training script

        elif choice == "2":
            logging.info("Starting webcam face recognition...")
            import webcam_recognition
            webcam_recognition.run()  # Run the webcam face recognition script

        elif choice == "3":
            logging.info("Exiting...")
            break
        else:
            logging.warning("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
