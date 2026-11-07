# 동일 민원 데이터의 flat·계층·검색 방식 비교

기존 실행 결과를 같은 81개 단일 정답 민원 기준으로 다시 모아 대분류, 대+중 경로, 전체 대+중+소 경로 정확도를 계산한 사후 비교입니다. 새로운 모델 실행이나 새로운 benchmark가 아닙니다.

## 데이터

- 작성자 정의 가상 분류표: 대분류 3 × 중분류 3 × 소분류 3 = 27 leaf
- 대분류: 생활환경, 도로·교통, 행정·복지
- 단일 정답 81건: 명시적 27, 암시적 27, 혼합 문맥 27
- 추가 확인 필요 9건: 정확도에서 제외
- 공식 지자체 자료나 독립 검수 데이터가 아님

## 결론

| 모델·방식 | 대분류 | 대+중 | 전체 경로 |
| --- | ---: | ---: | ---: |
| Kanana flat-27 zero-shot | 97.5% | 95.1% | 92.6% |
| EmbeddingGemma 2 retrieval top-1 | 92.6% | 85.2% | 71.6% |
| LAYA flat-27 | 69.1% | 59.3% | 34.6% |
| LAYA hard cascade | 59.3% | 35.8% | 23.5% |
| LAYA predicted-prefix cascade | 59.3% | 37.0% | 23.5% |
| KoBERT `[MASK]` flat-27 | 33.3% | 14.8% | 6.2% |
| KoBERT `[MASK]` predicted-prefix cascade | 53.1% | 25.9% | 16.0% |

현재 fixture에서는 Kanana flat-27이 가장 좋다. 그러나 LAYA에서는 flat이 cascade보다 좋았고 KoBERT에서는 cascade-prefix가 flat보다 좋았다. 계층 구조의 효과는 모델과 학습 형식에 따라 달라졌으며, 모델 간 차이를 계층 구조 하나의 효과로 해석할 수 없다.

beam12는 정답 후보를 76.5%에서 유지했지만 새 12-way rerank 정확도는 8.6%였다. 후보 recall과 최종 top-1 정확도를 분리해야 한다.

## 재계산

```powershell
python research/compare_flat_hierarchy_methods.py
```

수치와 비교 한계는 `summary.json`에 보존한다.
