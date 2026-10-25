import os
import shutil
import unittest

import logger


class LoggerTest(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(logger.log_dir(True), ignore_errors=True)
        self.prod = logger.log_path("train", "9.9", test=False)
        self.prod_existed = os.path.exists(self.prod)

    def test_train_log_written(self):
        logger.update_train_log("all", "9.9", "ridge", 1.5, 2.0, 0.1, test=True)
        rows = logger.read_log("train", "9.9", test=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["model"], "ridge")
        self.assertEqual(float(rows[0]["rmse"]), 1.5)

    def test_predict_log_written_and_appended(self):
        logger.update_predict_log("singapore", "9.9", "2019-06-01", 100.0, 90.0, 0.01, test=True)
        logger.update_predict_log("all", "9.9", "2019-06-02", 200.0, 190.0, 0.02, test=True)
        rows = logger.read_log("predict", "9.9", test=True)
        self.assertEqual([r["country"] for r in rows], ["singapore", "all"])

    def test_test_logs_do_not_touch_production(self):
        logger.update_train_log("all", "9.9", "ridge", 1.0, 2.0, 0.1, test=True)
        self.assertEqual(os.path.exists(self.prod), self.prod_existed)
        self.assertIn(os.sep + "test", logger.log_path("train", "9.9", test=True))

    def test_read_missing_log_is_empty(self):
        self.assertEqual(logger.read_log("train", "0.0", test=True), [])

    def tearDown(self):
        shutil.rmtree(logger.log_dir(True), ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
