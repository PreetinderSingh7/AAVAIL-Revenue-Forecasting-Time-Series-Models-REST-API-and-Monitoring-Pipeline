import unittest

import app as app_module
from unittests import ensure_test_models


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_test_models()
        cls.client = app_module.app.test_client()

    def post(self, body):
        return self.client.post("/predict", json=body)

    def test_index(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("all", r.get_json()["countries"])

    def test_predict_specific_country(self):
        r = self.post({"country": "singapore", "year": 2019, "month": 6, "day": 1, "test": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["country"], "singapore")
        self.assertGreater(r.get_json()["y_pred"], 0)

    def test_predict_all_countries(self):
        r = self.post({"country": "all", "year": "2019", "month": "06", "day": "01", "test": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["country"], "all")

    def test_unknown_country_rejected(self):
        self.assertEqual(self.post({"country": "mars", "year": 2019, "month": 6, "day": 1, "test": True}).status_code, 400)

    def test_missing_fields_rejected(self):
        self.assertEqual(self.post({"country": "all"}).status_code, 400)

    def test_bad_date_rejected(self):
        self.assertEqual(self.post({"country": "all", "year": 2019, "month": 13, "day": 1, "test": True}).status_code, 400)

    def test_monitor_endpoint(self):
        r = self.client.get("/monitor?test=1")
        self.assertEqual(r.status_code, 200)
        self.assertIn("models", r.get_json())

    def test_logs_endpoint_rejects_bad_kind(self):
        self.assertEqual(self.client.get("/logs/other").status_code, 400)


if __name__ == "__main__":
    unittest.main()
