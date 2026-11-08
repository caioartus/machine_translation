"""Lightweight checks: no GPU, model downloads, or ML packages required."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import load_examples, prediction_records


class EvaluationIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "data.jsonl"
        self.rows = [{"id": "b", "zh": "你好", "en": "Hello"},
                     {"id": "a", "zh": "谢谢", "en": "Thank you"}]

    def save(self, rows):
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def test_limit_preserves_order_and_total(self):
        self.save(self.rows)
        rows, total = load_examples(self.path, 1)
        self.assertEqual(rows, self.rows[:1])
        self.assertEqual(total, 2)

    def test_duplicate_even_beyond_limit_rejected(self):
        self.save([self.rows[0], self.rows[0]])
        with self.assertRaises(ValueError):
            load_examples(self.path, 1)

    def test_missing_reference_rejected(self):
        self.save([{"id": "a", "zh": "你好"}])
        with self.assertRaises(ValueError):
            load_examples(self.path)

    def test_empty_dataset_rejected(self):
        self.save([])
        with self.assertRaises(ValueError):
            load_examples(self.path)

    def test_mismatched_predictions_rejected(self):
        with self.assertRaises(ValueError):
            prediction_records(self.rows, ["Hello"])

    def test_empty_prediction_kept_for_honest_scoring(self):
        records = prediction_records(self.rows, ["", "Thank you"])
        self.assertEqual([r["id"] for r in records], ["b", "a"])
        self.assertEqual(records[0]["prediction"], "")


if __name__ == "__main__":
    unittest.main()
