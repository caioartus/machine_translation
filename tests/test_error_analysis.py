"""Synthetic statistical/alignment tests: no GPU, model download or test-set tuning."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from error_analysis_utils import align_to, coverage_features, paired_interval
from score_sentences import read_hypotheses
from quantitative_errors import corpus_metrics, matched_baseline, paired_rows, read_runs, write_quantitative
from analyze_errors import write_annotation_pairs
from compare_annotations import compare, read_annotations, population_rates


def fixture():
    return pd.DataFrame({
        "id": ["a", "b", "c"], "zh": ["甲", "乙", "丙"],
        "ref": ["a good translation", "another nice translation", "a short text"],
        "hyp": ["a good translation", "another wrong translation", "a short text"],
        "src_len": [5, 20, 70], "tgt_len": [3, 3, 3], "rarity_mean": [7., 8., 8.],
        "length_bin": ["short", "mid_short", "long"],
        "rarity_bin": ["typical", "rare", "rare"],
        "cell": ["short/typical", "mid_short/rare", "long/rare"],
        "chrf": [50., 50., 50.], "bleu": [40., 40., 40.],
        "len_ratio": [1., 1., 1.], "missing_nums": [0, 0, 0],
        "empty": [False]*3, "has_cjk": [False]*3, "repeated": [False]*3})


class AnalysisTests(unittest.TestCase):
    def test_align_reorders_but_rejects_missing_extra_and_duplicate(self):
        df = fixture()
        self.assertEqual(align_to(df.iloc[::-1], df.id.tolist(), "fixture").id.tolist(), df.id.tolist())
        for invalid in [df.iloc[:2], pd.concat([df, df.iloc[:1]]), df.assign(id=["a", "b", "d"])]:
            with self.assertRaises(ValueError):
                align_to(invalid, df.id.tolist(), "fixture")

    def test_coverage_is_token_weighted_and_training_is_leave_one_out(self):
        class Tokenizer:
            def tokenize(self, text):
                return text.split()
        rows = [{"id": "x", "zh": "a a b z"}]
        counts = Counter(a=4, b=1)
        held = coverage_features(rows, counts, Tokenizer(), 2).iloc[0]
        self.assertEqual(held.n_unseen_alma, 1)
        self.assertEqual(held.n_lowfreq_alma, 1)
        self.assertEqual(held.unseen_ratio, .25)
        train = coverage_features([{"id":"x", "zh":"a a b"}], counts, Tokenizer(), 2, True).iloc[0]
        self.assertEqual(train.n_unseen_alma, 1)
        self.assertEqual(train.n_lowfreq_alma, 2)

    def test_jsonl_aligns_ids_and_preserves_empty_hypotheses(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hyps.jsonl"
            rows = [{"id":"a", "zh":"甲", "en":"A"}, {"id":"b", "zh":"乙", "en":"B"}]
            path.write_text('\n'.join(json.dumps(x) for x in [{"id":"b", "hyp":""}, {"id":"a", "hyp":"A"}]))
            self.assertEqual(read_hypotheses(path, rows)[0], ["A", ""])
            path.write_text(json.dumps({"id":"a", "hyp":"A"}) + '\n')
            with self.assertRaises(ValueError):
                read_hypotheses(path, rows)

    def test_paired_sign_shares_and_constant_bootstrap(self):
        base, after = fixture(), fixture()
        after["chrf"] = [60., 40., 50.]
        rows = paired_rows(after, base, 100, 7)
        overall = next(r for r in rows if r["group_type"] == "overall")
        self.assertEqual(overall["mean_delta"], 0.)
        for name in ["win_share", "loss_share", "tie_share"]:
            self.assertAlmostEqual(overall[name], 1/3)
        np.testing.assert_allclose(paired_interval([2., 2., 2.], 101, 0), [2., 2.])

    def test_corpus_metrics_are_recomputed_from_text(self):
        df = fixture()
        df["hyp"] = df.ref
        df["chrf"] = 5.
        m = corpus_metrics(df)
        self.assertAlmostEqual(m["corpus_chrf"], 100.)
        self.assertEqual(m["mean_sentence_chrf"], 5.)

    def test_paired_seed_matching_rejects_wrong_beam_or_seed(self):
        base = [("uniform", fixture(), {"seed":"0", "train_size":"full", "num_beams":"1", "condition":"uniform"})]
        meta = {"seed":"0", "train_size":"full", "num_beams":"4", "condition":"length"}
        with self.assertRaises(ValueError):
            matched_baseline(base, meta, "sampling")
        meta.update(condition="uniform")
        self.assertEqual(matched_baseline(base, meta, "decoding")[0], "uniform")

    def test_glob_cannot_mix_seeds_with_changed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (0, 1):
                p = Path(tmp) / f"uniform_nfull_seed{seed}" / "test.csv"
                p.parent.mkdir()
                df = fixture()
                if seed:
                    df.loc[0, "zh"] = "wrong source"
                df.to_csv(p, index=False)
            with self.assertRaises(ValueError):
                read_runs(["uniform_nfull_seed*"], "test", tmp)

    def test_annotation_labels_preserved_and_changed_text_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scored = root / "scored" / "uniform_nfull_seed0"
            scored.mkdir(parents=True)
            fixture().to_csv(scored / "test.csv", index=False)
            output = root / "labels.csv"
            args = (["uniform_nfull_seed0"], ["a"], "test", root / "scored", output)
            write_annotation_pairs(*args)
            labelled = pd.read_csv(output, keep_default_na=False)
            labelled.loc[0, ["error_type", "severity", "notes"]] = ["ALT", "0", "valid wording"]
            labelled.to_csv(output, index=False)
            write_annotation_pairs(*args)
            self.assertEqual(pd.read_csv(output).loc[0, "notes"], "valid wording")
            changed = fixture(); changed.loc[0, "hyp"] = "changed output"
            changed.to_csv(scored / "test.csv", index=False)
            with self.assertRaises(SystemExit):
                write_annotation_pairs(*args)

    def test_blind_annotation_key_and_paired_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = fixture().iloc[:1].copy()
            b = a.copy()
            a["system"], b["system"] = "baseline", "system"
            a["error_type"], b["error_type"] = "OMIT", "OK"
            a["severity"], b["severity"] = "2", "0"
            df = pd.concat([a, b], ignore_index=True)
            df["notes"] = ""; df["sample_id"] = ["A1", "A2"]; df["system_code"] = ["S1", "S2"]
            df[["sample_id", "id", "system_code", "system"]].to_csv(root / "key.csv", index=False)
            df.drop(columns="system").to_csv(root / "blind.csv", index=False)
            restored = read_annotations(root / "blind.csv", root / "key.csv")
            _, summary, types = compare(restored, "baseline", ["system"])
            r = summary[summary.group_type == "overall"].iloc[0]
            self.assertEqual(r.fixed, 1)
            self.assertEqual(r.error_share_before, 1.)
            self.assertEqual(r.error_share_after, 0.)

    def test_population_estimates_weight_cells_and_refuse_incomplete_labels(self):
        details = pd.DataFrame([
            {"id":"a", "baseline":"b", "system":"s", "cell":"short/typical", "errors_before":"OMIT", "errors_after":""},
            {"id":"c", "baseline":"b", "system":"s", "cell":"long/rare", "errors_before":"", "errors_after":"OMIT"}])
        summary = pd.DataFrame([
            {"baseline":"b", "system":"s", "group_type":"cell", "group":"short/typical", "n_paired":1, "n_sampled":1},
            {"baseline":"b", "system":"s", "group_type":"cell", "group":"long/rare", "n_paired":1, "n_sampled":1}])
        population = pd.DataFrame({"id":["a", "b", "c"], "src_len":[5, 5, 70], "rarity_mean":[7., 7., 8.]})
        r = population_rates(details, summary, population).iloc[0]
        self.assertAlmostEqual(r.weighted_error_share_before, 2/3)
        self.assertAlmostEqual(r.weighted_error_share_after, 1/3)
        wrong = details.copy()
        wrong.loc[0, "cell"] = "long/rare"
        with self.assertRaisesRegex(ValueError, "disagree"):
            population_rates(wrong, summary, population)
        summary.loc[0, "n_sampled"] = 2
        self.assertTrue(population_rates(details, summary, population).status.str.startswith("not_estimated").all())

    def test_full_report_keeps_exact_seed_pairs_and_real_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for condition in ["uniform", "length"]:
                for seed in (0, 1):
                    name = f"{condition}_nfull_seed{seed}"
                    path = root / "scored" / name / "test.csv"
                    path.parent.mkdir(parents=True)
                    df = fixture()
                    df["num_beams"] = 1
                    if condition == "length":
                        df["chrf"] += 2 if seed == 0 else 4
                    df.to_csv(path, index=False)
            out = root / "report"; out.mkdir()
            result = write_quantitative(["uniform_nfull_seed*", "length_nfull_seed*"],
                                        "uniform_nfull_seed*", "test", root / "scored", out, 20)
            paired = pd.read_csv(out / "paired_changes_by_run.csv")
            all_rows = paired[(paired.group_type == "overall") & (paired.metric == "chrf")]
            self.assertEqual(all_rows.mean_delta.tolist(), [2., 4.])
            self.assertEqual(all_rows.baseline_run.tolist(), ["uniform_nfull_seed0", "uniform_nfull_seed1"])
            agg = pd.read_csv(out / "paired_changes_across_seeds.csv")
            overall = agg[(agg.group_type == "overall") & (agg.metric == "chrf")].iloc[0]
            self.assertEqual(overall.mean_delta, 3.)
            self.assertAlmostEqual(overall.seed_delta_sd, np.sqrt(2))
            self.assertEqual(len(list(out.glob("delta_chrf_*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
