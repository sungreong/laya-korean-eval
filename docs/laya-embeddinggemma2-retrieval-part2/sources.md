# Sources — LAYA 한국어 민원 분류 2편

조사·실험일: 2026-10-07

## 1차 출처

- Google, [EmbeddingGemma 2 model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2)
  - 740M 전체 구성, 270M text-only 구성, 24 layers, 768d output, MRL, 8K context, prompt와 공식 benchmark에 사용.
- Google, [EmbeddingGemma 2 launch](https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/)
  - 출시 배경과 intended use 확인.
- Google, [Fine-tune EmbeddingGemma 2 with Sentence Transformers](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers)
  - text-only selective loading, triplet 구성, `SentenceTransformerTrainer`, `MultipleNegativesRankingLoss`, BF16 학습 예제 확인.
- Google, [Fine-tune Multimodal Embeddings with EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-multimodal-embeddinggemma-with-sentence-transformers)
  - column-specific task prompt와 cross-modal triplet 학습 방법 확인.
- Google/Hugging Face, [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)
  - 실행 checkpoint와 SentenceTransformers 예제 확인.
- Google AI Edge, [Decision Maker guide](https://developers.google.com/edge/mediapipe/solutions/decision/decision_maker)
  - bi-encoder와 cross-encoder decision 구조 비교 참고.
- Google AI Edge, [Prompting and schema design best practices](https://developers.google.com/edge/mediapipe/solutions/decision/decision_maker/best-practices)
  - 선택지 설명 설계 참고.
- Sentence Transformers, [Samplers](https://sbert.net/docs/package_reference/sentence_transformer/sampler.html)
  - in-batch negative 학습에서 `NO_DUPLICATES` sampler가 필요한 조건 확인.
- Sentence Transformers, [Losses](https://sbert.net/docs/package_reference/sentence_transformer/losses.html)
  - `MultipleNegativesRankingLoss`, cached variant, Matryoshka loss의 입력 조건 확인.
- Sentence Transformers, [PEFT adapters](https://www.sbert.net/examples/sentence_transformer/training/peft/README.html)
  - `add_adapter()`를 이용한 LoRA 지원 범위 확인. EmbeddingGemma 2 민원 성능 근거로 사용하지 않음.

## 발견 경로

- [GeekNews topic 34907](https://news.hada.io/topic?id=34907)
  - 후속 실험 아이디어의 출발점. 모델 수치와 아키텍처는 공식 원문에서 재확인함.

## 직접 측정 자료

- [GitHub repository](https://github.com/sungreong/laya-korean-eval)
- `results/embeddinggemma2-retrieval-2026-10-07/`
  - 기존 27유형 retrieval, 36유형 회귀, 새 9유형 일반화, LAYA top-k reranking의 summary와 문항별 prediction.
- Model revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- 직접 측정값은 모두 합성 진단 세트 결과이며 공식 benchmark나 production 성능이 아님.
