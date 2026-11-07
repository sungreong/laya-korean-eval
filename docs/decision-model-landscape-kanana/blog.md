---
title: "Jev에서 Kanana-2 V-Decide까지: 두 달 만에 확장된 Decision Model 경쟁"
theme: report
intent: narrative
pageWidth: 1120px
toc: false
---

# Jev에서 Kanana-2 V-Decide까지: 두 달 만에 확장된 Decision Model 경쟁

> **최초 조사:** 2026년 10월 12일 · **Kanana 보강:** 2026년 11월 8일  
> **한 줄 정의:** Decision model은 글을 쓰는 모델이 아니라, 프로그램이 미리 정한 선택지를 읽고 각 선택지의 확률을 돌려주는 모델이다.

최근 본 [“Jev 벌써 끝? MS Decision-1 미쳤다”](https://youtube.com/shorts/EtWkF5Ocr8E)라는 Shorts는 과장이 섞인 제목처럼 보이지만, 시장의 속도만큼은 정확히 짚었다. TypeSafe AI가 Jev를 공개한 2026년 9월 15일부터 Microsoft가 Decision-1을 발표한 10월 9일까지 24일밖에 걸리지 않았다. 그 사이 LAYA, THIS/THAT, Typical, Clef, Strands Decider, H2O-Lightning과 OpenAI Decisions API가 연달아 등장했다.

11월 6일에는 Kakao가 [Kanana-2-v-decide-3b-2611](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611)을 공개했다. 한국어 전용 decision benchmark 수치를 제시하고 가중치와 구현을 함께 공개했다는 점에서, “한국어에서도 되는가”라는 질문을 직접 시험할 수 있는 첫 사례가 추가됐다.

이 글은 “어느 모델이 1등인가”를 선언하려는 글이 아니다. 공개된 아키텍처와 가격, 재현 가능한 자료, 아직 확인되지 않은 주장을 분리해 **무엇이 실제로 새롭고 어디에 쓸 만한지** 판단하는 것이 목적이다.

## BLUF: 먼저 결론

1. **Decision model의 핵심 가치는 정확도 단독 우승이 아니다.** 자유 텍스트 생성과 파싱을 없애고, 런타임 선택지에 대한 확률을 한 번의 scoring pass로 반환한다. 그래서 routing, 분류, 검증, 우선순위화처럼 호출량이 많은 지점에서 latency와 비용을 크게 줄일 수 있다.
2. **Jev가 시장의 이름과 API 계약을 대중화했고, 오픈소스가 내부 구현을 빠르게 다양화했다.** LAYA는 encoder와 option marker, Typical·Strands는 decoder torso와 readout/pointer head, Clef는 큰 Qwen backbone과 joint schema head를 택했다.
3. **상용 API 가격은 현재 100만 input tokens당 대략 0.038~0.24달러다.** 500-token 요청 1,000건이면 약 0.019~0.12달러다. 단, 네트워크·로그·검토·재시도 비용은 별도다.
4. **오픈 가중치는 무료 API가 아니다.** 라이선스 비용이 없을 뿐 GPU, 운영, batch 효율, idle time, 관측성 비용을 부담한다. H2O는 특정 측정 조건에서 1,000 decisions당 0.021달러를 추정했지만 모든 환경에 일반화할 수 없다.
5. **이 글의 동일 민원 데이터에서는 Kanana flat-27이 냉정하게 가장 낫다.** 81건에서 Kanana 92.6%, EmbeddingGemma 검색 71.6%, 계층 학습 LAYA flat-27 34.6%였다. 다만 서로 다른 크기·학습법의 모델이므로 “flat 방식이 항상 우월하다”거나 한국어 일반 성능이 92.6%라고 확대하면 안 된다.
6. **현재 권장 구조는 decision model 단독 자동화가 아니라 선택적 escalation이다.** 높은 확신은 바로 실행하고, 낮은 확신이나 고위험 항목은 강한 LLM 또는 사람에게 보낸다.

![2026년 9월부터 11월까지 주요 decision model 공개 타임라인](assets/timeline.svg)

## 무엇이 새롭고, 무엇은 원래 있던 기술인가

분류기는 오래전부터 있었다. reranker는 query와 후보를 함께 읽고 점수를 매겼고, LLM도 constrained decoding이나 JSON schema로 정해진 형식만 출력할 수 있었다. 확률 보정, selective prediction, learning to defer도 새로운 개념이 아니다.

이번 흐름이 주목받는 이유는 여러 기존 기술이 **하나의 제품 계약**으로 묶였기 때문이다.

- 호출자가 `state`, `question`, `options`를 보낸다.
- 모델은 선택지별 확률을 돌려준다.
- 자유 텍스트를 생성하지 않는다.
- `choice`, `score`, 참·거짓 확률을 같은 API에서 처리한다.
- 하나의 state에 여러 질문을 붙여 병렬 또는 shared-prefix 방식으로 처리한다.
- 확률 임계값에 따라 실행, 보류, 상위 모델 호출을 코드로 결정한다.

따라서 혁신의 중심은 “Transformer가 처음으로 판단했다”가 아니라 **생성 모델 중심의 인터페이스를 machine-native probability interface로 바꾼 것**에 가깝다. 2026년 9월의 [초기 증거 감사 논문](https://arxiv.org/abs/2609.32160)도 typed readout 자체가 같은 backbone의 label-probability 방식보다 독립적인 정확도 우위를 보였다고 말하기는 어렵고, 가장 분명한 이점은 latency와 비용이라고 정리한다.

## 시장은 어떻게 두 달 만에 커졌나

| 날짜 | 기업·팀 | 공개 내용 | 의미 |
| --- | --- | --- | --- |
| 09.15 | TypeSafe AI | Jev early access | `state + typed questions → probabilities`를 System One 모델로 제품화 |
| 09.18 | ConvAI Innovations | LAYA 공개 저장소 | ModernBERT/mmBERT encoder 기반의 소형 Apache-2.0 구현 |
| 09.20 | FLock.io·Oxford 연구진 | THIS/THAT 논문·가중치 | 1.88B 모델, 30.9ms local decision 측정 공개 |
| 09.24 | Oz Labs | Typical small/medium | truncated decoder, KV-cache 공유, explicit abstention 공개 |
| 10.01 | Cloudflare | Clef·Clef-flash | 27B·9B open weights와 Workers AI 호스팅, joint schema head 공개 |
| 10.01 전후 | Strands Labs | Strands Decider 2B | Qwen torso의 LM head를 pointer head로 교체한 Apache-2.0 구현 |
| 10.02~07 | H2O.ai | H2O-Lightning-4B | 4B multimodal decision model과 LoRA·vLLM 경로 공개 |
| 10.06 | OpenAI | Decisions API beta | GPT-6 Luna 기반 text+image typed decision endpoint 공개 |
| 10.09 | Microsoft | Microsoft-Decision-1 | Qwen3.5-9B post-training, Foundry·OpenRouter 호스팅 |
| 10.09~11 | Cloudflare | Clef-omni | 30B-A3B MoE로 audio·video까지 decision input 확대 |
| 11.06 | Kakao | Kanana-2-v-decide-3b-2611 | 3.67B multimodal open weights와 한국어 decision 평가 공개 |

타임라인에서 중요한 변화는 모델 수만 늘어난 것이 아니다. 처음에는 “Jev와 비슷한 공개 모델”이 중심이었지만, 10월에는 Cloudflare·OpenAI·Microsoft처럼 inference platform을 가진 기업들이 **호스팅 가격과 기업 운영 기능**을 붙였다. 11월의 Kanana는 다시 open weights 쪽에서 한국어와 vision, candidate embedding cache를 결합했다.

한편 [DecisionEval 디렉터리](https://decisioneval.dev/models/)는 9월 28일 기준 140개 모델을 기록했다. 이 숫자는 생태계의 속도를 보여주지만 품질을 뜻하지 않는다. 디렉터리 스스로도 대부분을 공급자 모델 카드만 확인한 community 항목으로 표시한다.

## 공통 아키텍처: 생성 loop를 scoring pass로 바꾼다

![상태와 질문이 backbone, option scoring head, 확률 정책을 거치는 공통 구조](assets/common-architecture.svg)

전통적인 생성 LLM은 `P(next token | prefix)`를 반복 계산해 문자열을 만든다. decision model은 질문별 선택지 집합을 먼저 알고 있으므로, 최종 단계에서 전체 vocabulary를 계속 sampling할 이유가 없다. 대신 선택지마다 logit을 만들고 질문 내부에서 softmax를 적용한다.

질문 (q)의 선택지가 (o_1, ..., o_K)이고 scoring head가 (z_i)를 만든다면 출력은 대체로 다음 형태다.

```text
p(o_i | state, q) = exp(z_i) / Σ_j exp(z_j)
```

모델마다 차이는 `z_i`를 만드는 방법에 있다.

### Encoder option marker: LAYA

[LAYA 모델 카드](https://huggingface.co/convaiinnovations/laya)는 ModernBERT-large 또는 mmBERT-base가 state와 질문·선택지를 함께 읽고, 각 선택지의 `[MASK]` marker 위치에서 점수를 뽑는다고 설명한다. 그 위에 두 개의 Transformer head layer와 option marker scorer, act/escalate head가 있다.

- English: ModernBERT-large, 전체 421M parameters
- Multilingual: mmBERT-base, 전체 322M parameters
- 장점: 작고 빠르며 encoder라 생성 loop가 없다. 다수 질문을 한 번에 처리한다.
- 단점: 기본 checkpoint의 zero-shot typed-decisions 성능이 낮다고 모델 카드가 직접 밝힌다. 50개 이상의 고 cardinality 선택지는 option token budget 부족으로 급격히 약해질 수 있다.
- 실제 의미: 범용 zero-shot 엔진보다 **도메인 fine-tuning용 기반 모델**에 가깝다.

### Decoder torso + readout: Typical, Strands, THIS/THAT

[Typical](https://typical.ozlabs.ai/)은 Qwen backbone을 약 71% 깊이에서 자르고, 아래쪽은 고정한 채 유지된 상위 layer에 rank-16 LoRA를 학습한다. state는 KV cache로 한 번 읽고, 질문과 선택지를 suffix로 붙인다. 작은 head가 질문의 final hidden state와 각 option representation을 비교한다. 별도 abstention head도 둔다.

[Strands Decider](https://github.com/strands-labs/strands-decider)는 Qwen3.5-2B-Base의 language-modeling head를 버리고 약 100만 parameter의 pointer head를 붙인다. `<answer>` 위치 hidden state와 각 선택지 마지막 token hidden state를 비교하며, rank-16 LoRA로 torso를 조정한다.

[THIS/THAT 논문](https://arxiv.org/abs/2609.23886)은 지정된 answer 위치 hidden state에서 선언된 선택지만 직접 읽는다. 1.88B parameters이며 모든 질문을 한 forward pass에서 처리한다.

- 장점: pretrained decoder의 지식을 활용하면서 decoding을 제거할 수 있다.
- 단점: 한 pass 안에서 중간 결과를 외부 memory처럼 순차 축적하기 어렵다. THIS/THAT은 multi-step arithmetic에서 0.560을 기록해 0.98~1.00 수준 comparator보다 약했다고 보고했다.
- 설계 포인트: option text가 실제 Transformer computation에 참여해야 새로운 label 설명으로 일반화할 가능성이 생긴다.

### Large backbone + joint schema head: Clef와 H2O

[Cloudflare Clef](https://blog.cloudflare.com/clef-decision-models/)는 Qwen3.8-27B, Clef-flash는 Qwen3.5-9B를 고정하고 rank-256 LoRA와 joint schema head를 학습한다. 선택지별 evidence routing, 질문 필드 사이의 cross-attention, 원래 state로 돌아가는 attention을 결합한 뒤 모든 option을 공동 scoring한다. label-smoothed cross-entropy와 Brier loss, RLCD를 사용했다고 설명한다.

[H2O-Lightning-4B](https://huggingface.co/h2oai/h2o-lightning-4b)는 Qwen3.5-4B 기반이며 LoRA adapter와 shim을 공개했다. text와 image를 처리하고 한 decision에 한 forward pass와 한 output token을 사용한다. base model과 adapter를 같은 vLLM server에서 함께 제공할 수도 있다.

- 장점: 큰 backbone과 multimodal 입력을 활용해 어려운 zero-shot 판단을 노린다.
- 단점: 9B~27B dense model은 LAYA보다 memory와 GPU 요구량이 크다. open weights여도 edge 배포가 가벼운 것은 아니다.

### Dual encoder와 cosine scoring: Kanana-2 V-Decide

Kanana-2 V-Decide는 state와 question을 합쳐 2,560차원 query embedding을 만들고, 각 candidate answer도 별도 2,560차원 embedding으로 만든다. 두 embedding의 cosine similarity를 온도 0.0561로 나눈 logit에 softmax를 적용한다.

```text
state + question ── Kanana-2 V ──> query embedding q
candidate text   ── Kanana-2 V ──> candidate embedding cᵢ

zᵢ = cosine(q, cᵢ) / temperature
pᵢ = softmax(z)ᵢ
```

기반은 Kanana-2-3B language model 3.18B와 vision encoder 0.49B를 합친 3.67B 모델이다. text, JSON, chat history와 최대 10개 image를 받을 수 있다. 모델 카드에 따르면 query와 candidate는 각각 별도 forward pass로 encoding하지만 candidate embedding은 token ID를 key로 cache한다. 따라서 처음 27개 후보를 계산하는 요청은 느리고, 같은 taxonomy를 반복하면 이후에는 state/question 쪽 계산만 남는다.

- 장점: runtime에 처음 보는 한국어 label 설명을 직접 비교하고, 동일 후보를 반복하는 routing에서 cache를 재사용할 수 있다.
- 단점: 한 요청에서 모든 option을 공동 attention하는 구조가 아니며, 후보가 자주 바뀌면 cache 이점이 줄어든다. 7.33GB BF16 weights와 Kanana Open License 조건도 운영 전에 검토해야 한다.

![Kanana-2 V-Decide의 dual-encoder 구조와 동일 민원 평가의 정확도 및 CPU 비용](assets/kanana2-v-decide-architecture-benchmark-ko.webp)

*그림 2. query와 candidate를 별도로 encoding해 cosine similarity로 비교한다. 아래 수치는 같은 합성 민원 81건에서 측정한 결과이며, 모델 크기와 CPU thread 조건은 서로 다르다.*

### Closed hosted model: Jev, OpenAI, Microsoft

Jev는 RLCD, parallel sampler, typed probability API를 공개했지만 parameters, backbone, scoring head는 공개하지 않았다. 따라서 외부에서 Jev의 내부 architecture를 LAYA나 Clef와 같다고 단정할 수 없다.

OpenAI Decisions API는 GPT-6 Luna로 `predicate`, `choice`, `score`를 지원한다. text와 image를 받을 수 있지만 전용 readout 내부 구조는 공개하지 않았다.

Microsoft-Decision-1은 Qwen3.5-9B를 single-pass decision scoring용으로 post-train했다고 밝혔다. 선택지별 calibrated probability를 반환하지만 세부 head, 학습 데이터, 공개 weights는 없다.

## 주요 모델 비교

| 모델 | 공개 형태 | 기반·크기 | 공개된 구조 | 잘 맞는 상황 | 주의점 |
| --- | --- | --- | --- | --- | --- |
| Jev | Hosted only | 비공개 | RLCD·parallel sampler 주장, 내부 비공개 | 빠른 상용 도입, 낮은 token 단가 | vendor lock-in, 자체 배포 불가, 내부 검증 제한 |
| LAYA | Open weights | ModernBERT 421M / mmBERT 322M | encoder + 2-layer head + option marker | 소형 self-host, 한국어 후보, 도메인 fine-tune | base zero-shot 약함, 많은 선택지에서 budget 문제 |
| THIS/THAT | Open weights | 1.88B | designated hidden-state readout | 저비용 local branch, edge | multi-step reasoning 약함, 초기 논문 표본 작음 |
| Typical | Open weights | 1.7B / 4B | truncated Qwen + KV reuse + readout + abstain | 한 state에 질문이 많을 때 | 별도 hosted 가격 없음, GPU 운영 필요 |
| Strands Decider | Open weights | Qwen3.5 1.9B | pointer head + rank-16 LoRA | agent routing, 소형 self-host | 초기 버전, benchmark 범위 제한 |
| Clef-flash | Hosted + open | Qwen3.5 9B | LoRA + joint schema head | 가격·품질 균형, vision 포함 | hosted context 24K, 9B 운영 부담 |
| Clef | Hosted + open | Qwen3.8 27B | joint attention schema scoring | 어려운 판단, vision·긴 문맥 | 높은 hosted 단가와 GPU 요구량 |
| H2O-Lightning | Open weights | Qwen3.5 4B | LoRA adapter + decision shim | text/image, base와 adapter 동시 serving | 9.1GB weights, 자체 운영 필요 |
| OpenAI Decisions | Hosted only | GPT-6 Luna, 비공개 | 전용 endpoint, 내부 비공개 | text+image, 기업 API·data controls | beta, $0.10/MTok, weights 비공개 |
| Microsoft-Decision-1 | Hosted only | Qwen3.5 9B post-train | single-pass scoring, 세부 비공개 | Foundry 기반 routing·grading | vendor benchmark 중심, self-host 불가 |
| Kanana-2 V-Decide | Open weights | Kanana LM 3.18B + vision 0.49B | query/candidate dual encoder + cosine softmax + candidate cache | 한국어 zero-shot routing, text·image 판단 | 7.33GB BF16, CPU 지연 큼, 자체 한국어 benchmark 비공개 |

## 가격: 100만 tokens보다 업무 한 건의 총비용을 보라

공개된 hosted 가격을 같은 가정으로 환산했다. **요청당 500 input tokens, 월 1,000만 decisions**이며 출력 token은 decision endpoint에서 무료라고 명시된 경우 0으로 계산했다.

| 서비스 | 공식 input 가격 / 1M tokens | 1,000건 비용 | 월 1,000만건 비용 |
| --- | ---: | ---: | ---: |
| Cloudflare Clef-flash | $0.038 | $0.019 | $190 |
| TypeSafe Jev | $0.042 | $0.021 | $210 |
| Microsoft-Decision-1 | $0.042 | $0.021 | $210 |
| OpenAI Decisions / GPT-6 Luna | $0.100 | $0.050 | $500 |
| Cloudflare Clef-omni | $0.150 | $0.075 | $750 |
| Cloudflare Clef | $0.240 | $0.120 | $1,200 |

단가만 보면 decision model은 매우 싸다. 그러나 production 비용은 다음 항목까지 합쳐야 한다.

- 입력 전처리와 OCR·speech-to-text
- 재시도와 provider fallback
- confidence가 낮은 요청의 LLM 또는 human review 비용
- 로그·감사·model version 추적
- prompt injection 검사와 개인정보 처리
- self-host일 경우 GPU idle time, autoscaling, container image, monitoring

오픈 모델은 API 요금이 없지만 공짜는 아니다. H2O는 자체 조건에서 H2O-Lightning-4B를 1,000 decisions당 0.021달러로 추정했고 9.1GB weights라고 밝혔다. THIS/THAT은 한 laptop GPU에서 30.9ms와 초당 32 decisions를 보고했다. Typical은 H100에서 cold state 한 건 45ms, 이후 질문당 2.7ms를 보고했다. 이 수치는 서로 다른 hardware·batch·질문 길이에서 측정됐으므로 표의 hosted 가격과 동일 선상 순위로 사용하면 안 된다.

## 성능표를 그대로 믿기 어려운 이유

Microsoft는 36개 benchmark, 약 15만 questions에서 가장 높은 정확도를 기록했고 GPT-6 Sol보다 p50 latency가 약 35배 빠르다고 발표했다. 8가지 perturbation에서 decision flip이 평균 1.3%였다는 robustness 결과도 공개했다. 하지만 평가셋 일부가 private이고 공급사 자체 측정이다.

Cloudflare도 Decision Index에서 Clef가 Jev보다 앞선다고 발표했다. H2O는 JevBench composite에서 H2O-Lightning-4B v1.1이 Jev보다 높다고 보고한다. Typical은 오히려 JevBench standard tier가 36개의 독립 state에서 파생된 72 items에 불과해 인접 모델의 차이를 분리하기 어렵다고 경고한다.

[JevBench coherence benchmark](https://jevbench.github.io/)는 12,000개의 변형 test로 paraphrase, option reorder, probability additivity, batch independence를 본다. 하지만 이 benchmark도 스스로 **일반 정확도나 속도를 측정하는 시험이 아니다**라고 명시한다.

따라서 최소한 다음을 분리해서 봐야 한다.

1. **Accuracy:** top-1 label이 맞는가
2. **Calibration:** confidence 0.9가 실제로 약 90% 맞는가
3. **Robustness:** option 순서·이름·문장 표현을 바꿔도 결론이 유지되는가
4. **Selective accuracy:** confidence threshold 아래를 넘겼을 때 자동 처리 구간의 정확도가 얼마인가
5. **Latency:** network 포함 p50·p95, batch 크기, state 길이가 같은가
6. **Total cost:** low-confidence escalation까지 포함했는가
7. **Baseline:** 같은 backbone의 label log-probability 또는 작은 classifier와 비교했는가

초기 evidence audit은 28개 연구 중 같은 backbone의 label-probability baseline을 포함한 연구가 26개 적용 대상 중 5개뿐이었다고 지적한다. 지금 단계에서 “typed head라서 더 정확하다”는 주장보다 “typed output이라 운영 계약이 단순하고 빠르다”는 주장이 더 잘 지지된다.

## 장점과 단점

### 장점

- **파싱 실패가 없다.** 출력 공간이 호출 전에 정의된다.
- **생성 token이 없다.** token-by-token decoding latency와 output 비용을 줄인다.
- **확률을 workflow에 직접 쓸 수 있다.** threshold, abstain, human review를 명시적으로 설계한다.
- **label이 runtime input이다.** 전통적인 고정 classification head보다 새로운 업무 유형을 설명으로 추가하기 쉽다.
- **한 state에 다수 질문을 붙이기 좋다.** 공통 state encoding 또는 prefix cache를 재사용할 수 있다.
- **오픈 구현 선택지가 빠르게 늘었다.** Apache-2.0·MIT 모델로 on-premise와 fine-tuning이 가능하다.

### 단점

- **선택지 밖의 답을 만들지 못한다.** 문제 정의가 틀리면 확신 있게 틀린 option을 고를 수 있다.
- **설명을 생성하지 않는다.** 규제·감사 업무에서는 evidence span, rule version, input snapshot을 별도로 저장해야 한다.
- **긴 multi-step reasoning에 약할 수 있다.** 한 pass가 빠른 이유와 같은 원리다.
- **확률은 자동으로 신뢰할 수 없다.** domain shift 뒤 temperature scaling과 threshold 재검증이 필요하다.
- **선택지가 수백 개면 비용과 구분 난도가 커진다.** 후보 retrieval 또는 계층 분류가 필요하다.
- **공개 benchmark가 아직 작고 중복이 많다.** 공급사별 1등 주장이 공존한다.

## Production 권장 구조

가장 현실적인 구조는 decision model이 모든 것을 대체하는 형태가 아니다.

```text
입력
  ├─ deterministic rule로 확정 가능한가? ── yes → 코드로 처리
  └─ no
      ├─ 후보가 수백 개인가? ── yes → embedding/BM25로 top-k 검색
      └─ decision model로 선택지별 확률 계산
          ├─ high confidence + low risk → 자동 실행
          ├─ medium confidence → 강한 LLM 재판단
          └─ low confidence 또는 high risk → 사람 검토
```

여기서 confidence threshold는 benchmark에서 가져오면 안 된다. false positive와 false negative의 실제 비용으로 정해야 한다. 예를 들어 민원 분류라면 잘못된 부서 자동 이관 비용, 재배정 시간, 긴급 민원 누락 비용을 각각 계산한 뒤 threshold를 고정하고 test split에서 평가해야 한다.

선택지가 300개라면 한 번에 300개를 넣기보다 다음 구조가 안정적이다.

1. embedding 또는 규칙으로 대분류·중분류 후보를 넓게 검색한다.
2. top-k leaf label의 이름과 설명을 decision model에 넣는다.
3. decision probability와 retrieval score를 별도 보존한다.
4. 두 점수가 충돌하거나 confidence가 낮으면 상위 모델로 보낸다.

이 구조는 LAYA 실험에서 확인했던 “잘 만든 encoder와 유형 설명이 있으면 다양한 label을 재학습 없이 활용할 수 있다”는 기대와도 맞닿아 있다. 다만 새로운 label을 설명만으로 처리하는 zero-shot 능력과 도메인 fine-tuning 성능은 별도로 측정해야 한다.

## 한국어에서는 무엇을 검증해야 하나

Kanana 공개 전에는 “Microsoft-Decision-1이 한국어에서도 1등” 또는 “LAYA multilingual이면 충분하다”고 판단할 자료가 없었다. Kanana는 공급사 내부 한국어 text 평가 15,909문항에서 72.50을 기록했다고 밝혔고, 비교한 다음 모델보다 18점 높다고 보고했다. 하지만 데이터와 문항별 결과가 공개되지 않아 독립 재현은 불가능하다. 따라서 아래 로컬 실험은 별도의 작은 진단으로 설계했다.

한국어 도입 평가에는 다음 항목이 필요하다.

- 조사·어미·높임말과 생략된 주어를 포함한 자연스러운 민원
- 한글 label 이름과 상세 설명, 영문 약어가 섞인 선택지
- 같은 의미를 다른 표현으로 바꾼 paraphrase 안정성
- option 순서 변경과 label key 변경에 대한 invariance
- 27개, 100개, 300개로 늘어날 때 top-1·top-k·latency 변화
- 대→중→소 prefix를 줄 때와 leaf 후보를 한 번에 줄 때의 차이
- ECE·Brier score와 threshold별 자동 처리율
- 새로운 유형을 설명만 추가한 zero-shot test와 few-shot fine-tuning test
- 긴 상담 요약과 개인정보 masking 이후의 성능

평가 결과는 전체 accuracy 하나로 끝내지 말고 다음처럼 보고하는 편이 낫다.

| 지표 | 묻는 질문 |
| --- | --- |
| Leaf top-1 accuracy | 최종 소분류를 바로 맞혔는가 |
| Hierarchical accuracy | 대·중·소 경로가 모두 맞는가 |
| Recall@k | 정답이 후보군 안에는 남았는가 |
| ECE / Brier | 확률을 threshold에 쓸 수 있는가 |
| Coverage@target accuracy | 95% 정확도를 유지하며 몇 %를 자동 처리할 수 있는가 |
| New-label accuracy | 학습에 없던 유형 설명으로 일반화하는가 |
| p50 / p95 latency | 실제 서비스 지연은 얼마인가 |
| Cost per 1K cases | escalation까지 포함한 비용은 얼마인가 |

## 로컬 CPU 한국어 민원 실험

### Methods

| 항목 | 설정 |
| --- | --- |
| 모델 | `kakaocorp/kanana-2-v-decide-3b-2611`, revision `1489062…`, BF16 |
| 장비 | Ryzen 7 PRO 7840U, Docker Desktop WSL2, CPU 8 thread, memory limit 11GiB |
| 데이터 | 저자 작성 합성 민원 90건: 단일 정답 81건 + 추가 확인 필요 9건 |
| 분류표 | 대분류 3 × 중분류 3 × 소분류 3 = leaf 27개 |
| 입력 | 상담 채널과 한국어 민원 요약 |
| 후보 | `대분류 > 중분류 > 소분류: 설명` 27개를 한 번에 제공 |
| 학습 | 이 분류표로 추가 학습하지 않은 zero-shot |
| 측정 | accuracy, Macro-F1, Wilson 95%, NLL, Brier, 10-bin ECE, p50/p95 latency |
| 지연 조건 | 첫 요청으로 candidate embedding cache를 채운 뒤 90건 측정 |

비교 대상은 같은 81건과 같은 27개 label을 사용한 기존 실행이다. 다만 LAYA와 EmbeddingGemma는 CPU 4 thread, Kanana는 8 thread였고 framework·cache 방식도 다르므로 latency는 통제된 모델 순위가 아니라 이 장비의 관측값이다.

### Results

| 모델·방식 | 정확도 | Macro-F1 | CPU p50 | 메모리·가중치 |
| --- | ---: | ---: | ---: | ---: |
| 기본 LAYA flat-27 | 24.7% (20/81) | 24.2% | 1.45초 | 322M급 |
| 민원 head 학습 LAYA flat-27 | 32.1% (26/81) | — | 1.61초 | 322M급 |
| EmbeddingGemma 2 256d 검색 top-1 | 71.6% (58/81) | — | 0.155초 | 측정 RSS 약 1.05GB |
| **Kanana-2 V-Decide zero-shot** | **92.6% (75/81)** | **92.1%** | **6.65초** | **warm RSS 6.29GB · weights 7.33GB** |

### 같은 자료를 대·중·소 단계로 다시 평가하면

이 평가 자료는 공식 지자체 데이터가 아니라 직접 만든 가상 민원이다. 대분류는 `생활환경`, `도로·교통`, `행정·복지` 3개이고, 각 대분류 아래 중분류 3개, 각 중분류 아래 소분류 3개를 둬 총 27개 leaf를 만들었다. 예를 들면 `도로·교통 > 도로시설 > 노면·보도 파손`처럼 전체 경로와 유형 설명이 붙는다.

전체 90건 중 정확도 계산에는 단일 정답이 있는 81건만 사용했다. 27개 유형마다 유형명이 직접 드러나는 문장, 간접 표현, 다른 단서가 섞인 문장을 하나씩 넣어 `명시적 27 + 암시적 27 + 혼합 문맥 27`로 구성했다. 어느 유형인지 추가 질문이 필요한 9건은 모델이 강제로 하나를 고르게 하되 정확도에서는 제외했다. 문장과 유형 설명을 같은 작성자가 만들었고 독립 검수도 없으므로, 실제 민원보다 경계가 선명할 수 있다.

평가 방식은 다음처럼 구분된다.

1. **일괄 flat-27:** `대 > 중 > 소: 설명` 27개를 한꺼번에 주고 최종 경로 하나를 바로 고른다.
2. **hard cascade:** 대분류 3개 중 하나를 고른 뒤, 그 아래 중분류 3개, 다시 소분류 3개를 순서대로 고른다. 앞 단계가 틀리면 정답 소분류는 다음 후보에서 사라진다.
3. **predicted-prefix cascade:** hard cascade와 같지만, 앞에서 고른 대·중분류 이름을 다음 단계 입력 prefix로 함께 제공한다. 학습 때는 정답 부모를, 실전에는 예측 부모를 쓴다.
4. **top-k beam12:** 대분류 top-2와 각 경로의 중분류 top-2를 유지해 4개 대·중 경로를 만들고, 그 아래 소분류 3개씩 총 12개를 최종 재평가한다.
5. **retrieval top-1:** 유형 설명 27개를 embedding으로 검색해 가장 가까운 leaf 하나를 고른다.

같은 81건의 최종 예측을 부모 경로까지 다시 계산하면 다음과 같다. `대+중`은 대분류와 중분류가 모두 맞은 비율이며, `전체 경로`는 대·중·소가 모두 맞은 비율이다.

| 모델·방식 | 대분류 | 대+중 경로 | 전체 대+중+소 | 냉정한 해석 |
| --- | ---: | ---: | ---: | --- |
| **Kanana flat-27 zero-shot** | **97.5%** | **95.1%** | **92.6% (75/81)** | 이 데이터에서 현재 최선. 다만 가장 느리고 cascade 미측정 |
| EmbeddingGemma 2 검색 top-1 | 92.6% | 85.2% | 71.6% (58/81) | 빠르고 강한 baseline. 최종 leaf 경계에서 오답 증가 |
| LAYA 계층 학습 flat-27 | 69.1% | 59.3% | 34.6% (28/81) | 같은 LAYA 순차 방식보다 낫다 |
| LAYA hard cascade | 59.3% | 35.8% | 23.5% (19/81) | 앞 단계 오류가 그대로 전파됐다 |
| LAYA predicted-prefix cascade | 59.3% | 37.0% | 23.5% (19/81) | prefix가 중간 경로를 조금 개선했지만 최종 정답 수는 그대로 |
| KoBERT `[MASK]` flat-27 | 33.3% | 14.8% | 6.2% (5/81) | 27개 전체 비교 형식에 거의 적응하지 못했다 |
| KoBERT `[MASK]` predicted-prefix | 53.1% | 25.9% | 16.0% (13/81) | 계층화로 상대 개선됐지만 절대 성능은 낮다 |

**현재 결과만 놓고 고르면 Kanana의 일괄 flat-27이 가장 낫다.** 그다음은 EmbeddingGemma 검색 단독이다. LAYA에서는 flat-27이 순차 분류보다 11.1%p 높았고, KoBERT에서는 반대로 predicted-prefix가 flat-27보다 9.9%p 높았다. 즉 계층 구조 자체가 성능을 보장하지 않는다. 모델이 학습한 후보 표현, 한 번에 비교하는 선택지 수, prefix의 학습·추론 일치 여부가 함께 맞아야 한다.

top-k도 자동으로 좋아지지 않았다. beam12는 정답 leaf를 62/81건(76.5%)에서 후보 안에 남겼지만, 학습하지 않은 12-way 전체 경로 재평가는 7/81(8.6%)만 맞혔다. 기존 flat-27 점수를 beam 후보에 제한하는 사후 계산도 26/81(32.1%)로 unrestricted flat-27의 28/81(34.6%)을 넘지 못했다. **후보 recall과 최종 top-1 정확도는 별개 지표**라는 뜻이다.

이 결과가 flat 방식을 보편적으로 추천한다는 뜻은 아니다. 현재는 leaf가 27개라 전체 후보를 한 번에 비교할 수 있었지만, 300개가 되면 입력 길이·후보 간 경계·추론 비용이 달라진다. 300개 환경에서는 retrieval로 넓게 top-k를 찾고, 그 후보 형식으로 따로 학습·검증한 decision scorer가 재평가하는 구조가 더 현실적이다. 이때도 `Recall@k`, `후보 내 조건부 정확도`, `최종 top-1`, `대·중·소 단계 정확도`를 각각 보고해야 한다.

또한 Kanana와 EmbeddingGemma에는 같은 fixture의 순차 cascade 실행이 없고, 모델 크기·학습 데이터·CPU thread·cache도 같지 않다. 따라서 이 표는 **현재 확보한 실행 결과에서 어느 조합이 잘 맞았는지**를 말할 뿐, flat과 cascade의 순수한 구조 효과를 통제한 실험은 아니다.

Kanana의 accuracy Wilson 95% 구간은 84.8~96.6%, p95 latency는 7.34초였다. NLL 0.228, multiclass Brier 0.097, 10-bin ECE 0.067을 기록했다. 정답 75건의 평균 최고 확률은 0.909, 오답 6건은 0.501이었다. 단, calibration split 없이 test set 자체에서 계산한 기술통계다.

같은 문항의 paired 비교에서 Kanana는 민원 head 학습 LAYA보다 +60.5%p, EmbeddingGemma 2 top-1보다 +21.0%p 높았다. EmbeddingGemma와 비교하면 Kanana만 맞힌 문항이 18개, EmbeddingGemma만 맞힌 문항이 1개였고 탐색적 exact McNemar p는 0.000076이었다. 이 검정은 한 번의 합성 데이터·단일 seed 결과이며 여러 비교 보정을 하지 않았다.

Cold start에서는 모델 객체 생성이 1.74초였지만 memory-mapped weights를 실제로 읽고 27개 candidate embedding을 만드는 첫 요청에 35.8초가 걸렸다. candidate cache가 채워진 두 번째 요청은 6.21초였다. 따라서 CPU batch service에는 쓸 수 있어도, 사용자 대화의 즉시 응답이나 scale-to-zero 환경에는 그대로 적용하기 어렵다.

### Discussion

가장 눈에 띄는 오답은 “비가 오면 물도 고이지만, 배수관보다 움푹 팬 아스팔트 구멍을 메워 달라”는 사례다. 정답은 `노면·보도 파손`이지만 Kanana는 `도로 배수`에 0.794를 줬다. `배수보다 포트홀`이라는 배제 표현보다 앞의 “물 고임” 단서를 강하게 잡은 것이다. 높은 전체 정확도와 별개로 negation·우선순위 문장은 별도 stress test가 필요하다.

92.6%가 높게 나온 이유는 세 가지로 해석할 수 있다. 첫째, Kanana가 한국어 decision training을 받은 모델이다. 둘째, query와 candidate 설명을 같은 embedding space에서 직접 맞추는 구조가 이 과제와 잘 맞는다. 셋째, 평가 문장과 유형 설명을 같은 작성자가 만들었기 때문에 실제 민원보다 어휘·의미 경계가 정돈돼 있다. 세 번째 요인은 성능을 크게 부풀릴 수 있다.

따라서 이 결과로 “Kanana의 한국어 실서비스 정확도가 92.6%”라고 결론 내릴 수 없다. 실제 도입 전에는 독립 상담원 라벨, 시간 분리 test, 기관별 표현, 개인정보 masking, 300개 label, 새 label holdout, option 순서 변경, image 입력과 confidence threshold를 다시 검증해야 한다. 추가 확인이 필요한 9건에서 허용 경로 9개를 모두 골랐지만 모델을 강제로 선택시킨 결과라 보류 능력을 측정한 것도 아니다.

## 어떤 모델을 선택할까

### 지금 바로 API로 검증하고 싶다

- 가장 낮은 공개 단가: Cloudflare Clef-flash
- Jev-compatible 상용 API: Jev, Microsoft-Decision-1, Clef 계열
- OpenAI 데이터·SDK 환경을 이미 쓴다: Decisions API
- Azure Foundry 거버넌스가 중요하다: Microsoft-Decision-1

### 내부 데이터가 외부로 나가면 안 된다

- 작은 footprint와 multilingual fine-tuning: LAYA multilingual
- decoder knowledge와 explicit abstention: Typical
- agent workflow와 작은 pointer model: Strands Decider
- vision과 비교적 강한 4B: H2O-Lightning-4B
- 한국어 zero-shot와 image를 함께 보며 6GB 이상 memory를 허용한다: Kanana-2 V-Decide
- 더 큰 multimodal quality를 감수할 GPU가 있다: Clef·Clef-omni

### label이 자주 늘어난다

런타임 label 설명을 읽는 모델이 유리하지만, 수백 label을 한 번에 넣는 것은 피한다. retrieval top-k와 decision scoring을 결합하고, 새 label은 별도 holdout으로 검증한다.

### 고위험 판단이다

모델 종류보다 정책이 중요하다. 자동 실행 구간을 좁게 시작하고, evidence와 confidence, model version, option schema version을 모두 기록하며, 낮은 confidence는 사람에게 보낸다.

## 결론: Kanana가 한국어 가능성을 보여줬지만 검증은 이제 시작이다

Microsoft-Decision-1의 등장은 Jev의 끝이라기보다 **decision model이 독립 제품군으로 인정받기 시작했다는 신호**에 가깝다. Kanana 공개는 여기에 한국어·multimodal·open weights라는 축을 더했다. 구현도 option marker, pointer head, joint schema head에서 query/candidate dual encoder까지 빠르게 갈라졌다.

현재 가장 강한 결론은 세 가지다.

1. **bounded decision에는 생성 LLM이 과한 경우가 많다.** parsing이 필요 없는 확률 출력은 실제 운영 이점이 있다.
2. **과제와 readout·평가 방식의 궁합이 중요하다.** 이 합성 민원에서는 candidate 설명을 별도로 encoding해 27개 전체 경로를 비교한 Kanana가 크게 앞섰다. LAYA는 flat-27이 cascade보다 나았고 KoBERT는 그 반대였으므로, 계층화만으로 개선된다고 볼 수 없다.
3. **승부는 단일 model benchmark보다 시스템 설계에서 난다.** retrieval, decision scoring, calibration, escalation, audit log를 함께 설계해야 한다.

한국어에서도 가능성은 충분하며, 이번 27-label zero-shot 결과는 그 기대를 이전보다 강하게 뒷받침한다. 잘 만든 유형 설명을 candidate encoder가 읽게 하면 고정 label classifier보다 taxonomy 변경에 유연할 수 있다는 가설도 합리적이다. 다만 300개 선택지, 독립 실데이터, 새 label, calibration과 비용은 아직 검증되지 않았다. 지금은 production 전체를 교체할 시점보다 **shadow traffic에서 Kanana·검색 모델·기존 분류기를 나란히 평가하고 자동 처리 구간을 찾을 시점**이다.

---

## 핵심 출처

- [TypeSafe AI — Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Microsoft — Introducing Microsoft-Decision-1](https://commandline.microsoft.com/microsoft-decision-1-model-foundry/)
- [OpenAI — Decisions API](https://developers.openai.com/api/docs/guides/decisions)
- [Cloudflare — Introducing Clef](https://blog.cloudflare.com/clef-decision-models/)
- [LAYA model card](https://huggingface.co/convaiinnovations/laya)
- [Oz Labs — Introducing Typical](https://typical.ozlabs.ai/)
- [FLock.io — THIS/THAT paper](https://arxiv.org/abs/2609.23886)
- [H2O-Lightning-4B model card](https://huggingface.co/h2oai/h2o-lightning-4b)
- [Kakao — Kanana-2-v-decide-3b-2611 model card](https://huggingface.co/kakaocorp/kanana-2-v-decide-3b-2611)
- [Strands Decider](https://github.com/strands-labs/strands-decider)
- [Typed Decision Models: An Early Evidence Audit](https://arxiv.org/abs/2609.32160)

