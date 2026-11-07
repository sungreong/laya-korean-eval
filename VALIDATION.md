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

## Kanana-2 V-Decide 동일 민원 평가

검증일: 2026-11-08 (Asia/Seoul). Docker Desktop Linux 컨테이너, CPU quota 8, 메모리 한도 11GiB.

- `kakaocorp/kanana-2-v-decide-3b-2611` revision `1489062def4145ae3a0a9261fbac4f4d88a8c6d1`을 별도 `kanana-cpu` 서비스에서 실행했다.
- 기존 27개 유형·90개 민원 fixture를 그대로 사용했다. 엄격 단일 정답 81건에서 75/81(92.6%), Macro-F1 92.1%를 기록했다.
- 같은 81건에서 EmbeddingGemma 2 검색 top-1은 58/81(71.6%), 민원 학습 LAYA는 26/81(32.1%), 기본 LAYA는 20/81(24.7%)였다.
- Kanana의 CPU p50은 6.65초, p95는 7.34초였다. 27개 후보 캐시를 만드는 첫 요청은 35.8초, cache-hit 요청은 6.21초, warm RSS는 약 6.29GB였다.
- 결과는 `results/kanana-2-v-decide-2026-11-08/`, 통합 기술 글은 `docs/decision-model-landscape-kanana/`에 보존했다.
- 이 수치는 단일 작성자가 만든 합성 표본의 탐색 실험이다. 모델별 스레드 수·프레임워크·캐시 조건도 완전히 같지 않아 일반적인 한국어 성능이나 운영 우위를 확정하지 않는다.
- 공개 결과의 문항 수·정답 수·동일 fixture 해시·세 비교군 수치를 검사하는 회귀 테스트 2개를 추가했다. 전체 단위 테스트 29개가 통과했다.
- 기존 예측을 다시 읽어 flat-27, hard cascade, predicted-prefix cascade, retrieval top-1의 대분류·대+중·전체 경로 정확도를 동일 81건 기준으로 계산했다. beam12의 후보 생존율과 최종 top-1 정확도도 분리해 `results/method-comparison-2026-11-08/`에 기록했다.
- 재계산된 주요 정답 수와 beam recall/top-1 분리를 검사하는 테스트 2개를 추가했다. 전체 단위 테스트는 31개다.

### 공식 자료·운영·라이선스 검토 보강

- Kakao Tech 글, 모델 카드·설정, 공식 `server_kanana_v_decide.py`, Kanana Open License, Kakao의 멀티모달 임베딩 발표 영상을 교차 확인해 통합 기술 글을 보강했다.
- Kakao가 공개한 H200 약 42ms는 공급사 측 한국어 상품 리뷰 판단 1건 수치다. 로컬에는 사용 가능한 CUDA 환경이 없어 GPU 속도는 재현하지 않았으며, 로컬 CPU 6.65초와 직접적인 배속 비교를 하지 않았다.
- 공개 서버는 요청 queue와 API key를 제공하지만 모델 호출을 전역 lock으로 직렬화한다. 따라서 고처리량 production server가 아니라 참고 구현으로 평가했다.
- 공식 checkpoint는 fine-tuning 가능한 공개 가중치지만, 2026-11-08 확인 기준 공식 학습 script·loss·dataset schema·LoRA recipe는 제공되지 않았다. 글의 LoRA 경로는 일반 PEFT 자료와 공개 아키텍처를 바탕으로 한 실험 제안으로 표시했다.
- Kanana 계산 구조가 일반 embedding bi-encoder, Kanana V-Decide, cross-encoder reranker와 어떻게 다른지 별도 절로 구분했다. 앞선 LAYA 경로는 독립 인코딩형이며 이름만으로 cross-encoder reranker라고 간주하지 않았다.
- Kanana Open License 원문을 검토하고 필수 고지문을 `NOTICE-KANANA.txt`에 추가했다. 자체 서비스와 제3자 API·SI·on-device 재판매의 조건이 다르며, 법률 자문이 아니라 기술 검토임을 명시했다.
