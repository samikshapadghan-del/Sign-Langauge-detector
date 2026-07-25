import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_backend_modules_import_cleanly(self):
        modules = [
            "backend.features",
            "backend.app",
            "backend.detector",
            "backend.train_model",
            "backend.train_asl_from_webcam",
            "backend.collect_data",
        ]
        for name in modules:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
