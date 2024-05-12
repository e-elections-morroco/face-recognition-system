import unittest
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(parent_dir)

# Import functions from encoding_image module
from encoding_image import get_image_encoding, get_image_encoding_from_csv, extract_encodings, write_encodings_to_csv

class TestImageEncodingFunctions(unittest.TestCase):
    """
    Test cases for image encoding functions.
    """

    def test_get_image_encoding(self):
        """
        Test the get_image_encoding function.
        """
        # Provide the path to a test image
        image_path = Path("test/test_images/ouail.jpg")
        encoding = get_image_encoding(image_path)
        self.assertIsNotNone(encoding)
        self.assertIsInstance(encoding, np.ndarray)

    def test_get_image_encoding_from_csv(self):
        """
        Test the get_image_encoding_from_csv function.
        """
        # Provide the name of a test image and the path to a test CSV file
        image_name = "messi.jpg"
        csv_filename = Path("test/test_encodings.csv")
        encoding = get_image_encoding_from_csv(image_name, csv_filename)
        self.assertIsNotNone(encoding)
        self.assertIsInstance(encoding, np.ndarray)

    def test_extract_encodings(self):
        """
        Test the extract_encodings function.
        """
        # Provide the path to a folder containing test images
        image_folder = "test/test_images"
        encodings_dict = extract_encodings(Path(image_folder))
        self.assertIsNotNone(encodings_dict)
        self.assertIsInstance(encodings_dict, dict)

    def test_write_encodings_to_csv(self):
        """
        Test the write_encodings_to_csv function.
        """
        # Provide the path to a folder containing test images and the name of the test CSV file
        image_folder = "test/test_images"
        csv_filename = "test/test_encodings.csv"
        encodings_dict = extract_encodings(Path(image_folder))
        write_encodings_to_csv(encodings_dict, Path(csv_filename))
        self.assertTrue(os.path.exists(csv_filename))

if __name__ == '__main__':
    unittest.main()
