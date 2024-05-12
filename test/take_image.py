import unittest
import os
import sys
from pathlib import Path

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)
from take_image import detect_person_with_face_eyes_nose_mouth

class TestDetectPerson(unittest.TestCase):
    def test_valid_image(self):
        # Test with a valid image containing a person with face, eyes, nose, and mouth
        image_path = Path("./test/test_images/image_with_all_requirements.jpg")
        result = detect_person_with_face_eyes_nose_mouth(image_path)
        self.assertTrue(result)

    def test_invalid_image(self):
        # Test with an invalid image (file does not exist)
        image_path = Path("./test/test_images/unknown.jpg")
        result = detect_person_with_face_eyes_nose_mouth(image_path)
        self.assertFalse(result)

    def test_no_person(self):
        # Test with an image containing no person
        image_path = Path("./test/test_images/cat.jpg")
        result = detect_person_with_face_eyes_nose_mouth(image_path)
        self.assertFalse(result)

    def test_multiple_faces(self):
        # Test with an image containing multiple faces
        image_path = Path("./test/test_images/multiple_persone.jpg")
        result = detect_person_with_face_eyes_nose_mouth(image_path)
        self.assertFalse(result)

    def test_missing_features(self):
        # Test with an image containing a person with missing features (e.g., no eyes)
        image_path = Path("./test/test_images/hamass.jpg")
        result = detect_person_with_face_eyes_nose_mouth(image_path)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()