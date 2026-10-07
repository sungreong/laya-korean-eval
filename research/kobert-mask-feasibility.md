# KoBERT `[MASK]` 후보 점수 방식 검토

## 결론

기술적으로 가능하다. KoBERT는 각 토큰의 마지막 은닉 벡터를 반환하며 `[MASK]`도 768차원 벡터를 갖는다. 입력에 후보마다 `[MASK]`를 하나씩 두고 해당 위치를 `gather`한 다음, 모든 후보에 동일한 `768→1` scorer를 적용하면 후보 수가 달라져도 점수를 만들 수 있다.

하지만 KoBERT 자체가 이 점수의 의미를 이미 알고 있는 것은 아니다. KoBERT 사전학습의 `[MASK]` 목표는 가려진 어휘 복원이다. `[MASK]` 위치를 “후보 설명의 대표 벡터”로 사용하고 하나의 scorer로 민원 적합도를 계산하려면, scorer와 필요하면 KoBERT까지 민원 선택 데이터로 학습해야 한다. 학습하지 않은 무작위 scorer의 출력은 성능 결과가 아니다.

## 가능한 계산

후보가 27개이면 입력을 다음처럼 만든다.

```text
[CLS] 분류 지시문 [SEP] 민원 요약 [SEP]
[MASK] 생활쓰레기 수거 설명 [SEP]
[MASK] 불법투기 설명 [SEP]
...
[MASK] 공공시설 운영 불편 설명 [SEP]
```

KoBERT 출력은 `H ∈ R^(L×768)`이다. `[MASK]` 위치 27개를 모으면 `M ∈ R^(27×768)`, 공유 scorer `f: R^768→R`를 적용하면 `z ∈ R^27`이 된다. `softmax(z)`로 27개 후보 확률을 계산하고 argmax를 고른다. 후보를 28개로 늘리면 scorer의 파라미터 크기는 그대로이고 `M`의 행만 28개가 된다.

```python
marker_hidden = hidden.gather(1, marker_positions[..., None].expand(-1, -1, 768))
logits = shared_scorer(marker_hidden).squeeze(-1)
probabilities = logits.softmax(-1)
```

실제 형태를 보여주는 [예제 코드](../examples/kobert_mask_scorer.py)를 추가했다. 이 코드는 의도적으로 무작위 scorer를 사용하므로 학습 전 logits을 분류 결과로 해석하면 안 된다.

## 두 가지 서로 다른 `[MASK]` 사용법

1. **MLM 어휘 확률 사용**: `[MASK]` 자리에 “도로”, “복지” 같은 단어가 나올 확률을 비교한다. 별도 head 없이 시험할 수 있지만 라벨이 tokenizer의 한 토큰이어야 유리하고, 여러 토큰으로 된 설명·동의어·27개 계층 경로를 공정하게 비교하기 어렵다. label verbalizer 선택에 매우 민감하다.
2. **은닉 벡터 + 공유 scorer**: `[MASK]`의 768차원 문맥 벡터를 후보 대표값으로 쓰고 같은 scorer를 적용한다. 여러 토큰 후보 설명과 동적 후보 수를 처리할 수 있어 LAYA 방식과 더 가깝다. 대신 해당 scorer를 반드시 학습해야 한다.

민원 유형에는 두 번째 방식이 더 적합하다. “노면·보도 파손” 같은 라벨은 단일 어휘 예측 문제가 아니며, 설명 전체와 민원 요약의 의미 관계를 학습해야 하기 때문이다.

## LAYA와의 구조 차이

| 항목 | KoBERT에 직접 구현 | 현재 LAYA Multilingual |
|---|---|---|
| 기반 인코더 | 한국어 BERT, 12층·hidden 768·12 heads | mmBERT/ModernBERT 계열, 22층·hidden 768·12 heads |
| 사전학습 범위 | 한국어 Wikipedia 중심 | 다국어 대규모 사전학습 |
| 후보 readout | 새로 구현하고 학습해야 함 | `[MASK]` 후보 readout과 decision head가 이미 구현됨 |
| 최대 위치 | 512 | 설정상 최대 8,192, 기본 LAYA 호출 예산은 별도 설정 |
| attention | 12층 전체 dense attention | mmBERT local/global attention + LAYA의 추가 dense Transformer head |
| 후보 추가 | 입력을 다시 인코딩하고 동일 scorer 적용 | 입력을 다시 인코딩하고 동일 scorer 적용 |
| 한국어 장점 가능성 | 한국어 전용 tokenizer·pretraining | 여러 언어와 혼합 입력, 이미 decision 학습된 공개 체크포인트 |

KoBERT 공식 저장소는 12개 층, 768 hidden, 3,072 FFN, 12 heads, 최대 길이 512, SentencePiece vocabulary 8,002, 약 92M parameters를 명시한다. 공식 README의 NSMC 90.1%는 NSMC에 맞춰 fine-tune한 고정 분류 실험이다. 현재 LAYA의 zero/few-shot 후보 설명 방식과 직접 비교할 수 있는 수치가 아니다.

## 계산량과 제한

후보 설명을 한 시퀀스에 넣으면 전체 길이 `L = 상태 + 지시문 + 후보 설명 합계`가 된다. KoBERT의 dense self-attention은 층마다 대략 `O(L²d)`이므로 후보를 늘리면 attention 비용이 이차적으로 증가한다. 512-token 제한 때문에 27개 후보 설명과 긴 민원 요약을 동시에 넣으면 상태나 설명이 잘릴 가능성이 크다.

대안은 다음과 같다.

- 대→중→소 단계마다 후보 3개만 넣어 길이를 줄인다. 상위 오분류가 아래 단계로 전파된다.
- `(민원, 후보 설명)` 쌍을 후보별로 따로 인코딩한다. 길이 제한은 관리하기 쉽지만 후보 27개면 KoBERT를 27번 실행한다.
- bi-encoder로 민원과 후보를 별도 임베딩한 뒤 dot product를 계산한다. 후보 임베딩을 캐시할 수 있어 빠르지만 상호 attention이 없어 미세한 맥락 구분이 약해질 수 있다.
- 한국어 전용 encoder에 LAYA의 추가 head 구조를 이식하고 민원 데이터로 학습한다. 가장 직접적인 비교지만 새로운 모델 학습 실험이다.

## 한국어 특화 모델이면 자동으로 좋아지는가

보장되지 않는다. 한국어 tokenizer와 한국어 사전학습은 한국어 표현에 유리할 가능성이 있지만, 실제 성능은 다음이 함께 결정한다.

- 후보 설명을 읽고 비교하도록 학습했는지
- 민원 도메인 표현과 라벨 정의가 학습 자료에 포함됐는지
- 512-token 안에 후보 27개와 요약이 보존되는지
- 후보 순서·라벨 문자열·부정 및 다중 이슈에 강한지
- 고정 27분류가 목표인지, 운영 중 새로운 후보를 추가해야 하는지

고정된 27개 민원 분류만 필요하면 KoBERT의 `[CLS]` 벡터 뒤에 `Linear(768, 27)`을 붙여 supervised fine-tuning하는 방식이 더 단순하고 보통 효율적이다. 추론 시 새 카테고리를 설명만 추가해 다루고 싶을 때 `[MASK]` 후보 + 공유 scorer 구조의 가치가 생긴다.

## 공정한 후속 실험

같은 학습/검증/시험 분할로 다음 세 모델을 비교해야 한다.

1. KoBERT `[CLS] → 27` 고정 분류기
2. KoBERT `[MASK] 후보 → 공유 scorer` 동적 분류기
3. LAYA의 기존 후보 scorer

정확도·Macro F1뿐 아니라 새 후보 추가, 후보 순서 shuffle, 긴 입력 truncation, 대→중→소 오류 전파, CPU p50/p95 latency와 peak memory를 측정해야 한다. 이 비교를 실행하기 전에는 “한국어 특화라서 LAYA보다 낫다”고 결론낼 수 없다.

## 이번 실측 결과

위 2번의 최소 구현을 실제로 학습했다. `skt/kobert-base-v1` encoder는 동결하고, 후보마다 실제 `[MASK]`(token ID 4)를 삽입해 마지막 hidden vector를 `LayerNorm → Linear(768,768) → GELU → Linear(768,1)` 공유 scorer로 변환했다. 학습 729개, validation 243개, 5 epoch, head learning rate 1e-3, seed 42다.

27개 후보 원문은 574 token으로 512 제한을 넘었다. 민원 본문에 최대 128 token을 예약하고 각 후보에 같은 예산을 주자 후보당 14 token, 전체 약 498 token이 됐다. validation에서 입력 본문이 잘린 사례는 없었지만 후보 설명은 epoch당 총 27,600 token이 잘렸다.

학습 전 validation 6.2%에서 1 epoch 7.8%로 올랐고, 이후 5.8%, 7.4%, 5.3%, 5.8%였다. NLL은 3.296에서 3.190으로 완만하게 낮아졌지만 정확도는 안정적으로 개선되지 않았다. validation으로 선택한 1 epoch 모델의 독립 시험 정확도는 27개 일괄 4.9%(4/81), 대→중→소 순차 6.2%(5/81)였다. 같은 시험에서 민원 학습 LAYA는 각각 32.1%, 23.5%였다.

따라서 “KoBERT에서도 실제 `[MASK]`를 후보 marker로 써서 가변 개수 logit을 계산할 수 있는가”에는 **예**, “그 구조와 작은 합성 자료만으로 LAYA의 decision 능력을 재현하는가”에는 **아니오**가 이번 결과다. 이 차이를 한국어 encoder 자체의 열세로 해석할 수는 없다. encoder를 동결했고, KoBERT는 이 readout 목표로 사전학습되지 않았으며, 후보 설명을 잘랐기 때문이다.

## 1차 출처

- [SKTBrain KoBERT 공식 저장소](https://github.com/SKTBrain/KoBERT): 구조, 사전, 한국어 Wikipedia 학습 자료, NSMC fine-tuning 결과, Apache-2.0.
- [SK Telecom의 Hugging Face KoBERT 체크포인트](https://huggingface.co/skt/kobert-base-v1): Transformers `AutoTokenizer`/`AutoModel` 사용 예와 공개 가중치.
