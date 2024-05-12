import unittest
import os
import sys
from pathlib import Path

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

from compare_image import compare_faces, compare_face_use_csv_encoding

class TestFaceComparison(unittest.TestCase):
    """
    Unit tests for face comparison functions.
    """

    def setUp(self):
        """
        Set up test data.
        """
        self.known_image_path = Path("test/test_images/ouail.jpg")
        self.unknown_image_path = Path("test/test_images/messi.jpg")
        self.csv_filename = Path("test/test_encodings.csv")

    def test_compare_faces(self):
        """
        Test the compare_faces function.
        """
        result = compare_faces(self.known_image_path, self.unknown_image_path)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bool)
        self.assertEqual(result, False)

    def test_compare_face_use_csv_encoding(self):
        """
        Test the compare_face_use_csv_encoding function.
        """
        result = compare_face_use_csv_encoding("ouail.jpg", self.unknown_image_path, self.csv_filename)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bool)
        self.assertEqual(result, False)


if __name__ == '__main__':
    unittest.main()
