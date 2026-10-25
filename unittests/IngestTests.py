import unittest

import pandas as pd

import ingest


class IngestTest(unittest.TestCase):
    def test_clean_drops_missing_and_duplicates(self):
        row = dict(country="Singapore", invoice="1", price=10, times_viewed=2, year=2019, month=1, day=1)
        df = pd.DataFrame([row, row, dict(row, invoice="2", price=None), dict(row, invoice="3", country="United Kingdom")])
        out = ingest.clean(df)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out.country), {"singapore", "united_kingdom"})

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            ingest.clean(pd.DataFrame({"country": ["x"]}))

    def test_timeseries_has_all(self):
        df = ingest.clean(pd.DataFrame([dict(country="a", invoice="1", price=5, times_viewed=1, year=2019, month=1, day=d) for d in (1, 3)]))
        ts = ingest.build_timeseries(df)
        self.assertIn("all", ts)
        self.assertEqual(len(ts["a"]), 3)  # gap day filled with zeros


if __name__ == "__main__":
    unittest.main()
