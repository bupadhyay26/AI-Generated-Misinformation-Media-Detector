import io
import unittest
from PIL import Image
from media_detector import analyze_image_bytes

class TestMediaDetector(unittest.TestCase):
    def test_image_analysis(self):
        img = Image.new("RGB", (640, 480), "white")
        b = io.BytesIO(); img.save(b, format="JPEG")
        out = analyze_image_bytes(b.getvalue(), "sample.jpg")
        self.assertTrue(out["ok"])
        self.assertIn(out["label"], ("REAL", "AI-GENERATED"))
        self.assertEqual(out["width"], 640)
        self.assertEqual(out["height"], 480)

if __name__ == "__main__":
    unittest.main(verbosity=1)
