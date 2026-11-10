# Evaluation decoding verification

The evaluator explicitly sets do_sample=False,
num_return_sequences=1, length_penalty=1.0 and early_stopping=False,
matching compare_decoding.py. These settings are recorded in metadata.
Previously, it inherited early_stopping=True from the checkpoint.

Verification used Simon's shared uniform_nfull_seed0 checkpoint,
all 1,002 ALMA development examples, beam 4 and batch size 16.

Job 1097485 completed successfully. Both scripts ran within the same
NVIDIA H100 NVL MIG 3g.47gb allocation.

| Run | BLEU | chrF |
|---|---:|---:|
| Evaluator repeat 1 | 16.493746 | 46.898108 |
| Evaluator repeat 2 | 16.490623 | 46.896318 |
| Comparison script, beam 4 | 16.491374 | 46.902372 |

IDs, sources and references matched across all outputs.
Evaluator repeats differed on 1/1002 translations; each differed from
the comparison script on 2/1002 translations.
A separate tokenizer check found identical encoded inputs for all
1,002 examples.

The scripts closely agree under matched conditions, but bitwise
reproducibility is not established. Earlier separate-run results were
not exactly reproduced; the complete cause remains unresolved.
These are development verification results, not official-test results.

## Shared outputs

Shared results root:
/mnt/scratch/t/teng/shared/mt-evaluation/development/

- same-gpu-check-20261108-234010/: full-development predictions,
  BLEU/chrF metrics and evaluator metadata; no COMET scores.
- final-uniform-full-seed0-dev20-20261108-223920/: 20-example
  predictions, BLEU/chrF, metadata and validated COMET scores.

Both shared copies were verified against the originals with diff -qr.
Read permissions are configured; teammate access has not yet been confirmed.

Original outputs remain under outputs/eval/ in Jake's cluster checkout.
Simon’s shared checkpoints remain under:
/mnt/scratch/s/simonwim/shared/runs/
