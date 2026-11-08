# Kanana 대→중→소 Greedy·Top-2·Final Rerank 계층 평가

`Kanana-2-v-decide-3b-2611`을 기존 3×3×3 민원 taxonomy와 동일한 단일 정답 81건에 적용해, flat-27과 단계별 예측을 비교한 Docker CPU 실험이다. 추가 확인이 필요한 9건도 강제 선택 trace에는 포함했지만 정확도에서는 제외했다.

## 결론

- **기본 설명 Top-2 branch 뒤에 최종 8개 경로를 Kanana가 직접 다시 비교하자 79/81, 97.5%로 올라갔다.** 누적 로그점수 방식 75/81보다 4건, flat-27 75/81보다 4건 높았다.
- 기본 Greedy cascade는 73/81, 90.1%였다. 처음 선택한 부모를 하나만 유지하면서 2건을 더 잃었다.
- Top-2는 정답 대분류를 81/81, 정답 대·중 경로와 leaf를 80/81에서 최종 8개 후보 안에 남겼다. 누적 로그확률은 그중 75건을 골랐지만, 직접 8-way final rerank는 79건을 골랐다.
- 범위·대표 예·제외 기준을 길게 붙인 보강 설명도 final rerank에서는 79/81로 같았지만 CPU p50이 14.97초에서 22.80초로 늘었다.
- 이 결과는 **27개 leaf에서 계층 탐색이 flat 성능을 유지할 수 있다는 작은 증거**다. 1,000개 유형의 성능을 측정한 결과는 아니다.

## 평가 방식

### 1. Greedy cascade

```text
민원 + 대분류 설명 3개 → 대 top-1
민원 + 선택 대분류 prefix + 중분류 설명 3개 → 중 top-1
민원 + 선택 대·중분류 prefix + 소분류 설명 3개 → 소 top-1
```

앞 단계가 틀리면 정답 leaf가 이후 후보에서 사라진다. 모델 호출은 문항당 3회다.

### 2. Top-2 branch

```text
대 top-2
  × 각 대분류 아래 중 top-2 = 대·중 경로 4개
  × 각 중분류 아래 소 top-2 = 최종 leaf 경로 8개
```

최종 선택은 별도 8-way reranker를 호출하지 않고 다음 누적 점수가 가장 큰 경로로 정했다.

```text
score(path)
= log P(대 | 민원)
 + log P(중 | 민원, 대 context)
 + log P(소 | 민원, 대·중 context)
```

문항당 모델 호출은 `1 + 2 + 4 = 7회`다. 서로 다른 후보 집합의 softmax 확률은 자동으로 calibration되지 않으므로, 이 합산식은 검증이 필요한 baseline이다.

### 3. Top-2 branch + final rerank

7회 탐색으로 만든 8개 전체 경로와 설명을 민원 원문과 함께 Kanana에 다시 넣고, 동일한 한 번의 choice 문제에서 최종 1개를 고르게 했다. 총 호출은 8회다. 마지막 호출이 후보를 같은 softmax 안에서 직접 비교하므로 서로 다른 단계의 확률을 더하는 문제를 피한다.

## 결과

| 설명·방식 | 대분류 | 대+중 경로 | 전체 경로 | Macro-F1 | CPU p50 | CPU p95 | 호출 수 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기존 flat-27 | 97.5% | 95.1% | **92.6% (75/81)** | 92.1% | 6.65초 | 7.34초 | 1 |
| 기본 Greedy cascade | 93.8% | 91.4% | 90.1% (73/81) | 89.9% | **4.82초** | 6.88초 | 3 |
| **기본 Top-2 branch** | 96.3% | 93.8% | **92.6% (75/81)** | **92.2%** | 12.02초 | 37.23초 | 7 |
| **기본 Top-2 + final rerank** | **98.8%** | **97.5%** | **97.5% (79/81)** | **97.5%** | 14.97초 | 40.10초 | 8 |
| 보강 Greedy cascade | 91.4% | 88.9% | 88.9% (72/81) | 87.9% | 8.64초 | 13.00초 | 3 |
| 보강 Top-2 branch | 96.3% | 92.6% | 91.4% (74/81) | 91.1% | 17.91초 | 25.67초 | 7 |
| 보강 Top-2 + final rerank | **98.8%** | **97.5%** | **97.5% (79/81)** | **97.5%** | 22.80초 | 30.48초 | 8 |

지연은 Ryzen 7 PRO 7840U, Docker CPU 8 thread, BF16, candidate cache를 사용한 이번 실행의 관측값이다. 기본 Top-2 p95에는 branch별 새로운 후보 설명을 처음 만나는 cache-miss성 지연이 포함되어 있다. GPU 결과로 일반화하지 않는다.

### Flat-27과 문항별로 비교하면

| 방식 | 둘 다 정답 | 계층만 정답 | flat만 정답 | 둘 다 오답 |
| --- | ---: | ---: | ---: | ---: |
| 기본 Greedy | 70 | 3 | 5 | 3 |
| 기본 Top-2 | 72 | 3 | 3 | 3 |
| 보강 Greedy | 67 | 5 | 8 | 1 |
| 보강 Top-2 | 71 | 3 | 4 | 3 |

Top-2와 flat은 전체 정답 수가 같지만 같은 문항을 맞힌 것은 아니다. 기본 Top-2는 flat 오답 3건을 구했고, flat 정답 3건을 새로 잃었다.

## 후보 생존과 최종 선택을 분리하면

| 단계 | 기본 Top-2 정답 생존 | 보강 Top-2 정답 생존 |
| --- | ---: | ---: |
| 대분류 top-2 | 81/81 | 81/81 |
| 대+중 경로 top-2 | 80/81 | 80/81 |
| 최종 leaf 8개 | 80/81 | 80/81 |
| 누적점수 top-1 정답 | 75/81 | 74/81 |
| final rerank top-1 정답 | **79/81** | **79/81** |

기본 Top-2 누적점수에서 실패한 6건 중 1건만 탐색 단계에서 정답 후보가 사라졌다. final rerank는 기존 오답 5건을 정답으로 바꿨지만 기존 정답 1건을 오답으로 바꿔 순증은 4건이었다. 정답이 최종 8개에 남은 80건 중 79건을 맞혔다. 따라서 병목은 대부분 탐색보다 최종 점수 결합이었으며, 직접 비교가 효과적이었다.

예를 들어 “구청 운영 주차 공간을 한 달 단위로 결제할 수 있는가”라는 문항은 정답 `도로·교통 > 주정차 > 주차시설 이용`이 최종 8개에 남았지만 누적점수에서는 오답이었다. final rerank는 전체 경로를 직접 비교해 이런 점수 결합 오류를 복구했다. 남은 오답 2건 중 1건은 정답 경로가 탐색에서 탈락했고, 1건은 정답이 남아 있었지만 `버스 운행`을 `처리 지연·공무원 응대`로 잘못 골랐다.

## 보강 설명은 왜 오히려 떨어졌나

별도 [`hierarchy-descriptions-enriched.json`](../../datasets/complaints/hierarchy-descriptions-enriched.json)에 다음 정보를 추가했다.

- 대분류: 업무 범위와 제외할 다른 대분류
- 중분류: 대표 leaf와 가까운 sibling의 경계
- 소분류: 기존 정의와 “배경보다 최종 해결 요청 우선” 규칙

하지만 설명이 길어지면서 부정·제외 표현이 늘었고, query와 candidate forward의 token 수도 증가했다. 이번 표본에서는 대분류 오류가 기본 Greedy 5건에서 보강 Greedy 7건으로 늘었다. **설명을 길게 쓰는 것과 잘 쓰는 것은 다르다.** 다음 실험은 짧은 양성 정의를 유지하고 혼동쌍을 별도 학습 hard negative로 제공하는 편이 낫다.

## 1,000개 이상 유형을 준비하려면

이번 27개 결과만으로 1,000개 정확도를 예측할 수 없다. 1,000개에서는 다음 데이터 구조와 평가가 필요하다.

1. **안정적인 ID와 taxonomy version:** 이름 변경과 조직 개편을 label ID 변경과 분리한다.
2. **짧은 양성 정의:** 각 대·중·소 label에 핵심 범위, 대표 요청 2~3개, 최종 처리 행동을 기록한다.
3. **혼동쌍 표:** sibling·타 branch에서 자주 헷갈리는 label과 구분 기준을 별도 구조화한다. 모든 제외 문장을 inference 설명에 넣지 않는다.
4. **단계별 학습 row:** 같은 민원으로 `대분류 질문`, `정답 대 context의 중분류 질문`, `정답 대·중 context의 소분류 질문`을 만든다.
5. **예측 prefix 학습:** 학습 때 정답 부모만 주는 teacher forcing뿐 아니라 잘못된 부모·top-2 branch도 일부 넣어 inference와 맞춘다.
6. **hard negative:** 같은 중분류 sibling, 다른 중분류의 의미상 가까운 유형, negation·혼합 문맥을 포함한다.
7. **분리된 test:** 문장 중복 제거, 시간·기관·상담원 기준 split, 학습에 없던 신규 label holdout을 둔다.
8. **단계별 metric:** 대 top-k recall, 중 경로 recall, leaf Recall@8, 후보 내 조건부 top-1, 전체 경로 정확도, calibration과 latency를 함께 본다.

실제 1,000개에서는 모든 leaf를 flat으로 넣기보다 `retrieval top-k → 계층 Top-2/Top-k → calibrated path scorer → 낮은 confidence 사람 검토`가 현실적이다. 이 실험은 Top-2가 27개에서 flat 정확도를 유지했다는 근거이지, 1,000개에서도 자동으로 유지된다는 보장은 아니다.

## 재현

```bash
docker compose run --rm kanana-cpu \
  python -u research/run_kanana_hierarchy.py \
  --out evaluation/kanana-2-v-decide-hierarchy-basic-cpu

docker compose run --rm kanana-cpu \
  python -u research/run_kanana_hierarchy.py \
  --out evaluation/kanana-2-v-decide-hierarchy-enriched-cpu \
  --description-profile datasets/complaints/hierarchy-descriptions-enriched.json

docker compose run --rm kanana-cpu \
  python -u research/run_kanana_final_rerank.py \
  --source evaluation/kanana-2-v-decide-hierarchy-basic-cpu/top2_branch_predictions.json \
  --out evaluation/kanana-2-v-decide-final-rerank-basic-cpu

python research/analyze_kanana_hierarchy.py
```

`--resume`을 사용하면 문항별로 저장된 JSON 다음부터 이어서 실행한다.

## 파일

- `comparison.json`: flat·기본·보강 방식의 지표와 paired 비교
- `basic-summary.json`, `enriched-summary.json`: 실행 환경과 전체 지표
- `*-predictions.json`: 문항별 stage trace, probabilities, 최종 branch
- `basic-final-rerank-*`, `enriched-final-rerank-*`: 최종 8개 경로를 한 번 더 직접 비교한 8번째 호출 결과
- `errors-*.json`: 오답과 Top-2 상위 branch
- `executed-run_kanana_hierarchy.py`: 실제 Docker 실행본
- `executed-run_kanana_final_rerank.py`: final rerank 실제 Docker 실행본

이 결과는 단일 작성자의 합성 taxonomy와 문항으로 수행한 탐색 실험이며, 공식 Kakao benchmark나 실서비스 정확도가 아니다.
