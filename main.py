import face_recognition
import cv2
from pathlib import Path

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
        known_image = cv2.imread(str(known_image_path))
        unknown_image = cv2.imread(str(unknown_image_path))
        
        if known_image is None or unknown_image is None:
            print("Error: Could not read the image file(s).")
            return None

        # Proceed with face encoding if images loaded successfully
        known_encoding = face_recognition.face_encodings(known_image)[0]
        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        return results[0]
    except IndexError:
        print("No faces detected in one of the images.")
        return None

# Usage
known_image_path = Path("ouail.jpg")
unknown_image_path = Path("messi.jpeg")
result = compare_faces(known_image_path, unknown_image_path)
if result is not None:
    print("Face match (ouail,messi):", result)


unknown_image_path = Path("unknown.jpeg")
result = compare_faces(known_image_path, unknown_image_path)
if result is not None:
    print("Face match (ouail,unknown):", result)
