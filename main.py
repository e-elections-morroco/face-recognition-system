import face_recognition
import cv2
from pathlib import Path
from embeddings import get_image_encoding


def compare_faces(known_image_path: Path, unknown_image_path: Path) -> bool | None:
    """
    Compare faces in two images.

    Args:
        known_image_path (Path): Path to the image file containing the known face.
        unknown_image_path (Path): Path to the image file containing the unknown face.

    Returns:
        bool | None: True if faces match, False if faces don't match, None if there's an error.

    Raises:
        None

    Examples:
        >>> known_image_path = Path("ouail.jpg")
        >>> unknown_image_path = Path("messi.jpeg")
        >>> result = compare_faces(known_image_path, unknown_image_path)
        >>> if result is not None:
        >>>     print("Face match:", result)
    """
    if not known_image_path.exists() or not unknown_image_path.exists():
        print("Error: One or both image files do not exist.")
        return None
    try:

        # Proceed with face encoding if images loaded successfully
        known_encoding = get_image_encoding(known_image_path)
        unknown_encoding = get_image_encoding(unknown_image_path)
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        return results[0]
    except IndexError:
        print("No faces detected in one of the images.")
        return None

# Usage
"""
known_image_path = Path("images/ouail.jpg")
unknown_image_path = Path("images/messi.jpeg")
result = compare_faces(known_image_path, unknown_image_path)
if result is not None:
    print("Face match (ouail,messi):", result)


unknown_image_path = Path("images/unknown.jpeg")
result = compare_faces(known_image_path, unknown_image_path)
if result is not None:
    print("Face match (ouail,unknown):", result)
"""

def detect_faces_in_video(known_faces_folder: Path):
    """
    Detect faces in real-time video stream and match them with known faces.

    Args:
        known_faces_folder (Path): Path to the folder containing known faces.

    Returns:
        None
    """
    # Load known face encodings
    known_face_encodings = {}
    for image_file in known_faces_folder.glob("*.jpg"):
        known_name = image_file.stem
        known_encoding = get_image_encoding(image_file)
        known_face_encodings[known_name] = known_encoding

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Compare each face encoding with known face encodings
            for known_name, known_encoding in known_face_encodings.items():
                match = compare_faces(known_encoding, face_encoding)
                if match:
                    print("Detected:", known_name)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    video_capture.release()
    cv2.destroyAllWindows()

# Usage
known_faces_folder = Path("images")
# detect_faces_in_video(known_faces_folder)


# Usage

face_embedding = get_image_encoding(Path("images/unknown.jpg"))
print(face_embedding.shape)





