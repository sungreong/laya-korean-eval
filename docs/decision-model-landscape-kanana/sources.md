# 출처 목록과 주장 매핑

최초 조사 기준일: 2026-10-12. Kanana 보강 기준일: 2026-11-08. 공식 문서와 공개 모델 카드를 우선했으며, 공급사 benchmark는 공급사의 주장으로 표시했다.

## 시작점

- [YouTube Shorts](https://youtube.com/shorts/EtWkF5Ocr8E) — 사용자 제공 링크
- YouTube oEmbed metadata — 제목과 채널 확인

## TypeSafe AI / Jev

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — 2026-09-15 공개, RLCD, parallel sampler, typed output, $0.042/MTok, output 무료
- [TypeSafe OpenAPI](https://api.typesafe.ai/redoc) — `/v1/systemone`, 모델 metadata, question contract
- [TypeSafe enterprise](https://typesafe.ai/enterprise) — data handling, ZDR, enterprise 운영 정보
- [Jack & Jill case study](https://typesafe.ai/blog/jack-jill-jev-case-study) — 실제 production 비용 사례. 공급사 case study이므로 독립 검증으로 취급하지 않음

## Microsoft

- [Microsoft-Decision-1 announcement](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/) — 2026-10-09, Qwen3.5-9B post-training, 36-benchmark·15만 questions, 가격
- [Microsoft Foundry announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-microsoft-decision-1-in-microsoft-foundry-for-decision-and-classific/4562742) — use case와 Foundry 제공
- [OpenRouter model listing](https://openrouter.ai/microsoft/microsoft-decision-1) — 배포 채널 확인

## OpenAI

- [Decisions API guide](https://developers.openai.com/api/docs/guides/decisions) — GPT-6 Luna, text/image, predicate·choice·score, $0.10/MTok, beta, data controls
- [OpenAI changelog](https://developers.openai.com/api/docs/changelog) — 2026-10-06 beta 공개
- [GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna) — model pricing과 modalities

## Cloudflare

- [Introducing Clef](https://blog.cloudflare.com/clef-decision-models/) — 2026-10-01, Qwen backbones, rank-256 LoRA, joint schema head, loss와 RLCD
- [Clef model card](https://huggingface.co/Cloudflare/clef) — 27B, Apache-2.0, model files, 입력 형식, 자체 결과
- [Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/) — Clef $0.24, Clef-flash $0.038, Clef-omni $0.15 per MTok
- [Clef Workers AI model page](https://developers.cloudflare.com/workers-ai/models/clef/) — context와 media token billing

## LAYA

- [LAYA model card](https://huggingface.co/convaiinnovations/laya) — 421M·322M, ModernBERT/mmBERT, option marker, RLCD, latency, honest limits
- [LAYA GitHub](https://github.com/NandhaKishorM/laya) — SDK, inference, fine-tuning
- [Fine-tuning guide](https://github.com/NandhaKishorM/laya/blob/main/docs/finetune.md) — encoder freeze와 full fine-tune

## Typical

- [Introducing Typical](https://typical.ozlabs.ai/) — 2026-09-24, 1.7B·4B, truncated backbone, KV cache, readout, abstention, latency, benchmark caveats
- [Typical weights](https://huggingface.co/OzLabs/typical-small) — Apache-2.0 model card
- [Typical source](https://github.com/OzLabsAI/typical) — 공개 코드

## THIS/THAT

- [THIS/THAT paper](https://arxiv.org/abs/2609.23886) — 1.88B, 30.9ms, architecture, cost·성능과 multi-step arithmetic 한계
- [THIS/THAT 1.2 model card](https://huggingface.co/flock-io/this-that-model-1.2) — MIT, weights와 config
- [FLock announcement](https://www.flock.io/blog/flockios-this--that-model-is-for-high-speed-low-cost-decision-making) — 회사 발표

## H2O.ai

- [H2O-Lightning-4B model card](https://huggingface.co/h2oai/h2o-lightning-4b) — Qwen3.5-4B, Apache-2.0, LoRA, vLLM, hardware·memory·비용 추정, calibration threshold

## Kakao / Kanana-2-v-decide

- [Kakao Tech — AI 에이전트를 위한 멀티모달 의사결정 모델 공개](https://tech.kakao.com/posts/846) — Kakao 공식 공개 소개
- [Kakao Tech 영상 — “그 사진 찾아줘” 한마디면 끝: Kanana 멀티모달 임베딩 모델 개발기](https://www.youtube.com/watch?v=2crSurA7R-U) — Kanana-v-embedding의 대규모 학습, GPU memory 제약, RAG·상품 검색·추천 연동 배경. V-Decide의 직접 학습 recipe로 취급하지 않음
- [Kakao 공식 보도자료](https://www.kakaocorp.com/page/detail/12193?lang=KOR) — H200 1장 평균 0.04초 이내, 한국어 15,909문항, 웹 검색·상품 카테고리 내부 적용 계획과 상업 활용 가능 라이선스라는 공급사 설명
- [Kanana-2-v-decide collection](https://huggingface.co/collections/kakaocorp/kanana-2-v-decide) — 2026-11-06 공개된 모델 컬렉션
- [Kanana-2-v-decide-3b-2611 model card](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611) — 3.67B 구성, cosine-similarity decision architecture, 입력 형식, Kanana Open License, 공급사 benchmark
- [공개 config](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/blob/main/config.json)와 [모델 구현](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/blob/main/modeling_kanana_v_decide.py) — 32-layer text config, 27-layer vision config, last-token pooling, L2 normalization, 4,096-entry candidate LRU cache와 cosine logit 구현 확인
- [공식 reference server](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/blob/main/server_kanana_v_decide.py) — System One API, 단일 inference lock, localhost 기본 bind, API key 옵션
- [Kanana Open License](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611/blob/main/LICENSE) — 자사 서비스, 파생 모델·재배포 조건과 별도 commercial license가 필요한 API·SI·on-device 판매 범위
- 공급사 모델 카드의 `Kanana Decision Bench 72.50`, `Decision Index 61.34`, `Decision Index Vision 61.12`는 Kakao가 보고한 값이다. 한국어 내부 평가셋 15,909문항은 공개 데이터가 아니므로 독립 재현 결과로 취급하지 않는다.
- 로컬 CPU 평가는 `laya-korean-eval/research/run_kanana_complaints.py`와 `evaluation/kanana-2-v-decide-flat27-cpu-*`에 별도로 기록한다. 이 평가는 저자 작성 합성 민원 90건 진단이며 공식 benchmark와 직접 비교하지 않는다.

## Fine-tuning 참고

- [Hugging Face PEFT custom models](https://huggingface.co/docs/peft/developer_guides/custom_models) — 지원 목록에 없는 custom architecture도 target module을 명시해 LoRA를 적용하는 일반 지침
- [Hugging Face PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora) — `target_modules`, adapter 저장과 custom head 보존에 관한 공식 문서
- Kanana model repository에는 2026-11-08 기준 inference·server 코드는 있지만 공식 training script, dataset schema, LoRA target과 재현 recipe는 확인되지 않았다. 본문의 학습 경로는 공식 recipe가 아니라 공개 forward 구조와 PEFT 문서를 결합한 제안이다.

## Strands

- [Strands Decider repository](https://github.com/strands-labs/strands-decider) — Apache-2.0, Qwen3.5-2B, pointer head, rank-16 LoRA, latency
- [Architecture document](https://github.com/strands-labs/strands-decider/blob/main/docs/architecture.md) — option scoring 구조와 학습 목표

## 시장·평가·비판

- [DecisionEval model directory](https://decisioneval.dev/models/) — 140개 모델 snapshot, evidence grade
- [Typed Decision Models: An Early Evidence Audit and Evaluation Checklist](https://arxiv.org/abs/2609.32160) — 28개 초기 연구 검토, baseline·calibration·통계 한계
- [JevBench coherence tests](https://jevbench.github.io/) — 50개 metamorphic relation, coherence 범위와 한계
- [Hugging Face typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) — 400 cases, 2,000 decisions, accuracy·KL·Brier·ECE

## 비용 계산

Hosted API 비용은 각 공식 가격표를 사용했다. 500-token 요청 1,000건은 0.5 MTok이므로 `공식 단가 × 0.5`, 월 1,000만건은 5,000 MTok이므로 `공식 단가 × 5,000`으로 계산했다. 세금, 지역 premium, network, storage, review, retry는 제외했다.

