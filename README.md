
# Face Recognition System

![Face Recognition System](./docs/image-a.png)

## Folder Descriptions

- **compare_image.py**: Script for comparing faces in images.
- **encoding_image.py**: Script for extracting face encodings from images.
- **encodings.csv**: CSV file containing face encodings for known images.
- **images**: Directory containing sample images for testing.
- **LICENSE**: License file (e.g., MIT License).
- **main.py**: Main script for running face recognition tasks.
- **README.md**: README file (you are here).
- **requirements.txt**: File listing required Python packages.
- **test**: Directory containing test scripts and data.
  - **compare_image.py**: Test script for comparing faces.
  - **encoding_image.py**: Test script for encoding images.
  - **test_encodings.csv**: CSV file containing test face encodings.
  - **test_images**: Directory containing test images.

## Overview

This repository contains a Python-based face recognition system that utilizes computer vision and machine learning techniques to identify faces in images and real-time video streams. The system is built using the `face_recognition` library along with OpenCV for image processing and face detection.

## Features

- **Face Comparison**: Compare faces in two images to determine if they match.
- **CSV-Based Comparison**: Compare faces in an image with known faces stored in a CSV file containing image encodings.
- **Real-Time Face Detection**: Detect faces in real-time video streams and match them against known faces.
- **Encoding Extraction**: Extract face encodings from images and store them for future comparison.

## Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/e-elections-morroco/face-recognition-system.git
    ```

2. Navigate to the project directory:

    ```bash
    cd face-recognition-system
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script to perform face recognition tasks:

    ```bash
    python main.py
    ```

2. Follow the on-screen instructions to select the desired operation (e.g., face comparison, real-time face detection).

## Contributing

Contributions to this project are welcome! If you have any ideas, suggestions, or bug fixes, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by the need for a robust and efficient face recognition system.
- Special thanks to the contributors and maintainers of the `face_recognition` and OpenCV libraries for their invaluable work.


