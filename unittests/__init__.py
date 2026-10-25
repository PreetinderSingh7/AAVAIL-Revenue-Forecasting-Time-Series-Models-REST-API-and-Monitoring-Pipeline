import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_scores = None


def ensure_test_models():
    """Train models once per test run, strictly in test mode (models/test, logs/test, data/test)."""
    global _scores
    if _scores is None:
        import model
        _scores = model.train(test=True)
    return _scores
