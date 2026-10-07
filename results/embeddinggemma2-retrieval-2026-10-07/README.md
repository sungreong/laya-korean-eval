# EmbeddingGemma 2 retrieval + LAYA reranking — 2026-10-07

This folder publishes the exact JSON artifacts used in the second Korean technical blog post.

## Main findings

| Evaluation | Method | Accuracy | Recall@3 | Recall@8 |
| --- | --- | ---: | ---: | ---: |
| Existing 27 labels, 81 strict cases | EmbeddingGemma 2 256d retrieval | 58/81 (71.6%) | 91.4% | 100% |
| Existing 27 labels, 81 strict cases | 256d top-3 + LAYA rerank | 41/81 (50.6%) | 91.4% | — |
| Existing cases with 36-label index | EmbeddingGemma 2 256d retrieval | 53/81 (65.4%) | 90.1% | 100% |
| 9 unseen labels, 27 cases | EmbeddingGemma 2 256d retrieval | 16/27 (59.3%) | 88.9% | 100% |
| 9 unseen labels, 27 cases | 256d top-3 + LAYA rerank | 15/27 (55.6%) | 88.9% | — |

The retriever was not fine-tuned. The LAYA reranker reused the previously selected complaint checkpoint. This is a synthetic, author-defined diagnostic without independent human annotation.

## Layout

- `retrieval-main27`: 27-label index and existing complaint cases.
- `retrieval-main36`: 36-label index and the same existing complaint cases.
- `retrieval-unseen36`: 36-label index and 27 cases from nine labels excluded from LAYA training.
- `rerank-*`: LAYA predictions over the retrieved candidates.
- `manifest.json`: copied artifact inventory.

Every run stores its summary, per-case predictions, input hashes, model revision, and the executed script snapshot.

## Reproduce

```powershell
docker compose --profile setup run --rm prepare-retrieval
docker compose run --rm retrieval python -u research/run_embedding_retrieval.py `
  --model evaluation/models/embeddinggemma-2-914f7f89142e `
  --taxonomy datasets/complaints/taxonomy.json `
  --cases datasets/complaints/cases.json `
  --out evaluation/my-retrieval-run
```

Use a fresh output directory. The scripts refuse to overwrite prior results.
