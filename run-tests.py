"""Run every unit test (API, model, logging, ingestion) with one command: python run-tests.py"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

if __name__ == "__main__":
    import model
    suite = unittest.TestLoader().discover("unittests", pattern="*Tests.py", top_level_dir=ROOT)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    model.clean_test_artifacts()  # remove models/test, logs/test, data/test
    sys.exit(0 if result.wasSuccessful() else 1)
