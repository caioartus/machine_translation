# Greedy and beam search comparison

Evaluated the same nine full-data checkpoints: uniform, length, and joint,
each with seeds 0, 1, and 2, from runs_my_full/.

Training settings: learning rate 1e-5, 2900 steps, batch size 16,
and sampling cap 3.0. Checkpoints were selected by minimum development loss.

Evaluation uses ALMA development data, batch size 16, deterministic generation,
and num_beams 1 versus 4. No additional training was performed.
These are development results, not final test-set results.

## Mean scores across three seeds

| Strategy | Greedy BLEU | Beam 4 BLEU | Greedy chrF | Beam 4 chrF |
|---|---:|---:|---:|---:|
| uniform | 16.2241 | 16.4204 | 46.3254 | 46.8495 |
| length | 15.9336 | 16.5496 | 46.0731 | 46.8922 |
| joint | 15.8071 | 16.2610 | 45.8778 | 46.5452 |

Beam 4 improved both metrics for all nine checkpoints.
Length balancing had the largest average decoding gain.
Its small mean advantage over uniform under beam 4 is descriptive;
statistical significance has not been established.

## Reproduction

Place checkpoint directories under runs_my_full/, prepare ALMA development data,
and run from the repository root:

    sbatch jobs/compare-decoding.sbatch

The job activates the local CS4248 Conda environment.
Per-run metrics include metric signatures, GPU, package version, and elapsed
generation time. Timing is indicative, not a controlled performance benchmark.
