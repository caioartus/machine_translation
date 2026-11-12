# Greedy vs beam search

The nine full-data checkpoints (uniform, length, joint × seeds 0, 1, 2) decoded on ALMA dev with
`num_beams=1` and `num_beams=4`. Development results, not test results.

| Condition | Greedy BLEU | Beam 4 BLEU | Greedy chrF | Beam 4 chrF |
|---|---:|---:|---:|---:|
| uniform | 16.22 | 16.42 | 46.33 | 46.85 |
| length | 15.93 | 16.55 | 46.07 | 46.89 |
| joint | 15.81 | 16.26 | 45.88 | 46.55 |

Mean over three seeds. Beam 4 improved both metrics for all nine checkpoints. Differences between
conditions are not tested for significance.

Reproduce (from the repository root):

```bash
RUNS_DIR=/path/to/runs sbatch jobs/compare-decoding.sbatch
```

Results go to `runs_decode_ablation/<job id>/`; plots in `notebooks/decoding.ipynb`.
