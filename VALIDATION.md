# 공개 전 실행 검증

검증일: 2026-10-07 (Asia/Seoul). Docker Desktop Linux 컨테이너, CPU quota 4, 메모리 한도 8GiB.

- `docker compose build`: 성공. 기존 의존성 빌드 캐시 사용.
- 고정 커밋의 LAYA 소스와 NSMC 데이터: 새 프로젝트에서 다운로드 및 준비 성공.
- 모델: 이미 받은 동일 revision의 가중치를 재사용하고 파일별 SHA256/크기 검증 성공. 이번 검증에서 대용량 가중치의 새 다운로드를 끝까지 수행한 것은 아님.
- `docker compose --profile test run --rm test`: 6개 테스트 모두 통과. 지표 경계 조건, 공개 지표 재계산, 최초 실행본 해시, 데이터 표본/분할 검증 포함.
- 실행·준비·예제·테스트 Python 파일의 구문 검사: 통과.
- `research/run_evaluation.py`: 기본 모델 평가 → 장문 진단 → 2 epoch 학습 → 학습 후 평가 → 보정 → 보정 후 평가 → 체크포인트 저장까지 완료. `status: complete` 확인.
- `research/analyze_results.py`: paired 분석 완료. 최초 파일럿과 동일한 정확도 차이 및 구간 확인.

| 조건 | 2026-10-05 파일럿 | 공개 전 재실행 |
|---|---:|---:|
| TF-IDF + LR | 140/200 | 140/200 |
| 기본 모델·한국어 질문 | 132/200 | 132/200 |
| 기본 모델·영어 질문 | 146/200 | 146/200 |
| 추가 학습 후 | 132/200 | 132/200 |
| 추가 학습·보정 후 | 132/200 | 132/200 |
| 자체 진단·학습 전/후 | 각각 16/24 | 각각 16/24 |

이 검증은 배포용 코드의 실행 가능성과 기존 관측 재현을 확인한 것이다. 새 독립 표본이나 다중 seed 평가가 아니다. 실행 중 준비 작업이 일부 겹쳤으므로 재실행 지연시간을 별도 성능 향상의 근거로 사용하지 않는다. FP16 저장 체크포인트를 다시 로드한 정확도와 사용자 업무 학습 예제의 학습 실행은 이 검증에 포함되지 않는다.

최초 공개 측정은 `results/pilot-2026-10-05/`에 보존했다. 새 실행의 전체 로그·결과·체크포인트는 로컬 `evaluation/`에 생성되며 Git에서 제외된다.

## 같은 날짜의 후속 실험

- 학습 표본 1,000개·5 epoch 실험을 완료했다. 검증으로 선택한 모델을 FP16 저장 후 다시 로드하여 시험했다. 전체 약 28.4분. 결과와 문항별 예측은 `results/scaling-2026-10-07/`에 공개했다. 모델 파일과 NSMC 리뷰 원문은 포함하지 않는다.
- 가상 민원 90개와 3×3×3 분류 체계를 작성했다. 기본 모델/감정 학습 모델 × 일괄/순차 분류의 네 조건을 실제 실행했다. 결과는 `results/complaints-2026-10-07/`에 보존했다.
- 유형 트리, 사례 구성, 입력 해시, 원 실행본 해시, 검증 기반 epoch 선택, 원 예측으로부터의 지표 재계산을 검사했다. Docker 테스트 총 10개가 통과했다.
- 위 `results/complaints-2026-10-07/` 비교 시점에는 민원 학습을 수행하지 않았다.

## 민원 도메인 학습 및 KoBERT `[MASK]` 실험

- 합성 학습 729개와 validation 243개를 생성했다. 27개 유형의 표본 수가 같고 기존 90개 holdout과 완전 동일한 문장이 없음을 검사했다.
- LAYA decision head를 5 epoch 학습했다. validation으로 5 epoch을 선택하고 FP16 체크포인트 저장·재로딩을 확인했다. 81개 단일 정답 holdout의 flat27 정확도는 기본 20/81에서 26/81로 증가했다.
- `skt/kobert-base-v1` revision `359874884642d748079d4dd4ff547f2cdbff67d6`를 준비했다. tokenizer의 실제 `[MASK]` 문자열과 token ID 4, 모든 후보 marker 위치의 ID를 확인했다.
- KoBERT encoder를 동결하고 실제 `[MASK]` hidden vector의 공유 MLP scorer를 5 epoch 학습했다. validation으로 1 epoch을 선택했다. holdout 정확도는 flat27 4/81, cascade 5/81이었다.
- 27개 후보 원문은 574 token으로 KoBERT의 512 한도를 넘었다. 동일 후보 예산으로 줄인 뒤 본문 잘림은 없었고 후보 설명 잘림은 결과에 기록했다.
- 원 PNG 그래프를 보존하고 2096×666 WebP quality 85를 README에 사용했다. 78,981 bytes에서 40,744 bytes로 48.41% 줄었으며 글자, 선, 데스크톱 폭을 육안 검사했다.
- `python -m unittest discover -s tests -v`와 `docker compose --profile test run --rm test`: 총 14개 테스트가 모두 통과했다. 전체 Python 구문 검사와 `git diff --check`도 통과했다.
