# LAYA Top-k beam12 후속 실험

최종 계층+prefix LAYA 체크포인트를 그대로 사용해 2026-10-07 Docker CPU 환경에서 다시 추론했다. 학습은 추가하지 않았다.

추론 절차:

1. 대분류 3개 중 Top-2 유지.
2. 각 대분류 아래 중분류 3개를 평가하고 부모별 Top-2 유지.
3. 남은 4개 대·중 경로 아래의 소분류 3개를 모두 합쳐 12개 후보 구성.
4. `대 > 중 > 소: 설명` 형식의 12개 전체 경로를 한 번에 재평가해 한 개 선택.

81개 단일 정답 합성 시험에서 최종 정확도는 7/81(8.6%)였다. 정답 소분류는 62/81(76.5%)에서 최종 12개 후보 안에 남았지만, 재평가가 그중 7건만 Top-1으로 선택했다. 기존 flat-27 확률을 동일한 12개 후보에 제한하는 사후 분석은 26/81(32.1%)로, 제한 없는 flat-27의 28/81(34.6%)을 넘지 못했다.

| 방식 | 정답 | 정확도 | p50 지연 |
| --- | ---: | ---: | ---: |
| flat-27 | 28/81 | 34.6% | 1.394초 |
| hard cascade | 19/81 | 23.5% | 0.849초 |
| predicted-prefix cascade | 19/81 | 23.5% | 0.907초 |
| **beam12 + full-path rerank** | **7/81** | **8.6%** | **2.059초** |

이 결과는 Top-k 자체가 무의미하다는 뜻이 아니다. 후보 recall과 최종 Top-1 정확도가 다른 지표임을 보여 준다. 이번 모델은 12개의 전체 경로 설명을 비교하는 형식으로 학습되지 않았고, 후보 수와 설명 형식 변화에 민감했다. beam/rerank 형식을 사용하려면 같은 형식의 학습 자료, validation에서 정한 stage-score 결합, 별도 reranker를 비교해야 한다.

- `laya/beam12_predictions.json`: 90건 전체 입력·후보 확률·단계별 trace
- `laya/summary.json`: 정확도, slice, 지연, Wilson 구간
- `laya/beam_analysis.json`: 후보 생존율과 flat-score 제한 사후 분석
- `laya/executed-run_complaints.py`: 실제 실행 코드 보존본

재현 명령:

```bash
docker compose run --rm evaluate python research/run_complaints.py \
  --model evaluation/complaint-hierarchical-prefix-e1/selected-model \
  --out evaluation/complaints-beam12-reproduction \
  --methods beam12

python research/analyze_beam12.py \
  --beam evaluation/complaints-beam12-reproduction/beam12_predictions.json \
  --flat results/hierarchical-prefix-2026-10-07/laya/test/flat27_predictions.json \
  --out evaluation/complaints-beam12-reproduction/beam_analysis.json
```
