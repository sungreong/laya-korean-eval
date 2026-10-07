# Complaint-domain training results

Measured on 2026-10-07 in the Docker Compose CPU environment: 4 CPU quota, 8 GiB memory, seed 42. The training set has 729 synthetic rows, validation has 243, and the held-out test fixture has 81 single-answer plus 9 under-specified cases.

| Model | Selected epoch | Validation accuracy | Holdout flat27 | Holdout cascade | Training and validation time |
|---|---:|---:|---:|---:|---:|
| LAYA decision head | 5 | 41.2% | 32.1% (26/81) | 23.5% (19/81) | 10,244 s |
| KoBERT real-`[MASK]` shared scorer | 1 | 7.8% | 4.9% (4/81) | 6.2% (5/81) | 4,796 s |

`laya/train/` and `kobert/train/` contain epoch summaries, validation predictions, and the exact executed training scripts. `laya/test/` and `kobert/test/` contain held-out predictions and exact executed evaluation scripts. `comparison.json` contains paired exploratory statistics against the previously published base LAYA predictions.

The KoBERT adapter used tokenizer mask ID 4 at every recorded candidate marker. Its 512-token limit required equal per-option truncation for the 27-way question. This is an experimental adapter, not an official KoBERT or LAYA benchmark.

The original `training-curves.png` is preserved. README files use `training-curves.webp` (quality 85), visually checked at 2096×666. It is 40,744 bytes versus 78,981 bytes for PNG, a 48.41% reduction.
