import gzip
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_news_commentary as news  # noqa: E402


class PrepareNewsCommentaryTest(unittest.TestCase):
    def test_cleaning_reasons_and_deterministic_sample(self):
        lines = [
            "Hello\t你好\n",
            "Hello again\t你好\n",
            "Repeat\t重复\n",
            "Repeat\t重复\n",
            "\t空\n",
            "English target\tEnglish source\n",
            "123\t中文\n",
            "Protected\t受保护。\n",
            "malformed\n",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "en-zh.tsv.gz"
            with gzip.open(input_path, "wt", encoding="utf-8") as handle:
                handle.writelines(lines)

            # Keep the test independent of a downloaded M2M-100 tokenizer.
            with patch.object(news, "n_tokens", side_effect=lambda text: len(text.split())):
                candidates, removed, counts = news.clean(
                    input_path,
                    protected_exact=set(),
                    protected_near={news.loose("受保护")},
                )

        self.assertEqual([row["zh"] for row in candidates], ["你好", "重复"])
        self.assertEqual(
            counts,
            {
                "duplicate_source": 1,
                "exact_duplicate": 1,
                "empty_side": 1,
                "missing_chinese": 1,
                "missing_english": 1,
                "protected_loose": 1,
                "malformed_tsv": 1,
            },
        )
        self.assertEqual(len(removed), 7)

        first = news.deterministic_sample(candidates, 2, seed=0)
        second = news.deterministic_sample(list(reversed(candidates)), 2, seed=0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
