# Kanana-2 V-Decide 한국어 민원 CPU 평가

`kakaocorp/kanana-2-v-decide-3b-2611`을 기존 27개 민원 유형·90건 fixture에 zero-shot으로 적용한 Docker CPU 실행 결과입니다.

## 결론

- 단일 정답 81건 중 75건을 맞혀 **accuracy 92.6%, Macro-F1 92.1%**를 기록했습니다.
- 동일 문항에서 민원 head 학습 LAYA 32.1%보다 +60.5%p, EmbeddingGemma 2 256d 검색 top-1 71.6%보다 +21.0%p 높았습니다.
- warm-cache 지연은 p50 6.65초, p95 7.34초였습니다. 첫 요청은 27개 candidate embedding 생성까지 포함해 35.8초였습니다.
- warm RSS는 약 6.29GB, BF16 weight 파일은 7,330,945,212 bytes입니다.
- 데이터는 한 작성자가 만든 합성 진단이므로 실서비스 정확도나 한국어 전체 성능으로 일반화할 수 없습니다.

## 평가 조건

| 항목 | 값 |
| --- | --- |
| Model revision | `1489062def4145ae3a0a9261fbac4f4d88a8c6d1` |
| Weight SHA-256 | `528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b` |
| Runtime | Docker Desktop WSL2, PyTorch 2.10.0 CPU, Transformers 5.5.3 |
| CPU | Ryzen 7 PRO 7840U, 8 threads |
| Memory limit | 11GiB |
| Dtype / attention | BF16 / SDPA |
| Candidates | `대 > 중 > 소: 설명` 형식의 27개 leaf |
| Training | 이 민원 taxonomy로 추가 학습하지 않음 |
| Cases | 단일 정답 81건 + 확인 필요 9건 |

## 동일 데이터 비교

| 모델·방식 | 정답 | 정확도 | CPU p50 |
| --- | ---: | ---: | ---: |
| 기본 LAYA flat-27 | 20/81 | 24.7% | 1.45초 |
| 민원 head 학습 LAYA flat-27 | 26/81 | 32.1% | 1.61초 |
| EmbeddingGemma 2 256d 검색 top-1 | 58/81 | 71.6% | 0.155초 |
| **Kanana-2 V-Decide zero-shot** | **75/81** | **92.6%** | **6.65초** |

Kanana와 EmbeddingGemma의 paired 결과는 Kanana만 정답 18건, EmbeddingGemma만 정답 1건이었습니다. 탐색적 exact McNemar p는 0.000076, paired bootstrap 95% 정확도 차이는 +12.3~+30.9%p입니다. 단일 합성 데이터에 대한 사후 분석이며 다중 비교 보정은 하지 않았습니다.

## 난이도별 결과

| Slice | 정확도 |
| --- | ---: |
| 명시형 | 100.0% (27/27) |
| 간접 표현 | 88.9% (24/27) |
| 복합 맥락 | 88.9% (24/27) |

NLL은 0.228, multiclass Brier는 0.097, 10-bin ECE는 0.067입니다. 정답 건의 평균 최고 확률은 0.909, 오답 건은 0.501이었습니다. 별도 calibration split 없이 test set에서 계산한 기술통계이므로 threshold를 확정하는 데 사용하면 안 됩니다.

## 특이 오답

> 비가 오면 물도 고이지만 배수관 점검보다 움푹 패인 아스팔트 구멍 자체를 메워 달라는 요청임.

정답은 `L13 노면·보도 파손`이지만 모델은 `L15 도로 배수`에 0.794, 정답에 0.187을 줬습니다. 배수는 부차적이라고 명시했는데도 앞부분의 “비·물 고임” 단서를 더 강하게 사용했습니다. negation과 요청 우선순위를 별도 stress test로 검증해야 합니다.

## 재현 명령

```bash
docker compose build kanana-cpu
docker compose --profile setup run --rm prepare-kanana
docker compose run --rm kanana-cpu
docker compose run --rm kanana-cpu python -u research/benchmark_kanana_cold.py
python research/analyze_kanana_results.py
```

`prepare-kanana`만 network를 사용합니다. `kanana-cpu`는 `network_mode: none`으로 실행합니다. 공식 snapshot 다운로드가 느린 경우에도 완료된 모델 폴더를 유지하면 다시 사용할 수 있습니다.

## 파일

- `summary.json`: 90건 실행 설정과 주요 지표
- `predictions.json`: 문항별 27개 확률, 예측, 지연
- `comparison.json`: calibration 기술통계와 paired 비교
- `errors.json`: 오답 6건과 top-3 확률
- `cold-summary.json`: 첫 요청과 candidate-cache hit 비교
- `executed-run_kanana_complaints.py`: 실제 실행본

이 결과는 공식 Kanana Decision Bench 재현이 아닙니다. 이미지 입력, 300개 label, option 순서 교란, 새 taxonomy holdout, 독립 상담원 검수와 production traffic은 평가하지 않았습니다.
