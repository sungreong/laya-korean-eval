# Hierarchical and predicted-prefix complaint experiment

This folder preserves the measured summaries and per-case predictions without model checkpoints.

- `laya/hierarchy-train`, `kobert/hierarchy-train`: balanced `flat27`, `major3`, `middle3`, and `leaf3` training.
- `laya/prefix-train`, `kobert/prefix-train`: one additional epoch with gold parent prefixes.
- `laya/test`, `kobert/test`: independent 81-case strict test plus 9 under-specified cases. `cascade_prefix` passes predicted parents.
- `laya/unseen`, `kobert/unseen`: 27 cases for nine labels absent from training.
- `laya/base-unseen`: base LAYA on the same unseen fixture.
- `comparison.json`: compact cross-run measurements and protocol notes.
- `hierarchical-prefix-results.png` and `.webp`: source and optimized chart.

Training uses gold parent labels as prefixes (teacher forcing). Evaluation never uses gold parents: the predicted major category is injected into the middle-stage input, and the predicted major and middle categories are injected into the leaf-stage input. A wrong parent therefore removes the correct child branch and propagates through the prefix.

The training, validation, main test, and unseen test texts are separate. These are author-created synthetic diagnostics with one seed and frozen encoders, so they are not production benchmarks.
