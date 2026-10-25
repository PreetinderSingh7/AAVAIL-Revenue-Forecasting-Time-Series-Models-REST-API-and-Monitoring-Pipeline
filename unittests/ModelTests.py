import os
import unittest

import model
from unittests import ensure_test_models


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores = ensure_test_models()

    def test_models_saved_in_test_dir(self):
        for c in self.scores:
            self.assertTrue(os.path.exists(model.model_file(c, test=True)))

    def test_isolated_from_production(self):
        self.assertNotEqual(model.model_dir(True), model.model_dir(False))
        self.assertTrue(model.model_dir(True).endswith("test"))

    def test_multiple_models_compared(self):
        s = self.scores["all"]
        self.assertIn("baseline", s)
        self.assertGreaterEqual(len([k for k in s if k != "baseline"]), 3)

    def test_predict_all_countries(self):
        r = model.model_predict("all", "2019-06-01", test=True)
        self.assertIsInstance(r["y_pred"], float)
        self.assertGreater(r["y_pred"], 0)

    def test_predict_single_country(self):
        r = model.model_predict("singapore", "2019-06-01", test=True)
        self.assertEqual(r["country"], "singapore")
        self.assertIn("baseline", r)

    def test_missing_model_raises(self):
        with self.assertRaises(FileNotFoundError):
            model.model_load("atlantis", test=True)

    def test_date_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            model.model_predict("all", "2030-01-01", test=True)


if __name__ == "__main__":
    unittest.main()
