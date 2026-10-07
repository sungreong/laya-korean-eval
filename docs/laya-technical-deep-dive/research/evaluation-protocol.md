# 한국어 파일럿 평가 사전 계획

이 계획은 모델 출력을 확인하기 전에 작성했다. 목적은 완전한 리더보드 재현이 아니라 한국어의 실제 동작과 소규모 추가 학습의 변화를 측정하는 것이다.

## 가설

- H1: Laya Multilingual의 한국어 영화 리뷰 이진 분류는 균형 표본의 50% 무작위 기준을 넘는다.
- H2: 동일 한국어 본문에 한국어 또는 영어 지시문·선택지 설명을 사용해도 정확도 차이가 작다. 차이와 paired disagreement를 보고하고 보편적 동등성을 주장하지 않는다.
- H3: 작은 한국어 자료로 헤드만 추가 학습하면 같은 도메인의 held-out 정확도가 개선된다. 성공 여부와 관계없이 최종 epoch를 보고한다.
- H4: 같은 학습이 다른 업무의 판단 성능까지 유지한다는 보장은 없다. 학습에 넣지 않은 자체 한국어 부서 분류 진단으로 회귀를 관찰한다.
- H5: 한국어 문서가 기본 1,024토큰 예산을 넘으면 후반의 정답 근거를 잃을 수 있다. 앞/뒤 위치와 1,024/2,048 입력 한도를 구분해 기록한다.

## 고정 조건

- 모델: convaiinnovations/laya-multilingual, HF revision 1720e3e3357cfe1e281542e223f8273b0890ca34.
- 코드: Laya 0.3.27, git 8a6e1328cce2460a0e5aa348ad465bb1b5821cd2.
- NSMC: 공식 train에서 학습 200개(각 100), 보정 80개(각 40); 공식 test에서 평가 200개(각 100). 빈 문서 제거, 정확히 같은 문서 중복 제거, train/test 간 문서 중복 배제. seed 42. 실제 선택 ID와 원본 SHA256 보존.
- 자체 진단: 사전에 작성한 24개 부서 분류 문항. 일반 표현, 부정, 대화 문맥, 혼합 언어, 전문용어, 번역투 각 4개. 작성자가 모델 출력을 보기 전에 정답 지정. 독립 인간 검수 없음. 벤치마크와 분리 보고.
- 언어 표현 비교: NSMC 200개에 동일 선택지 키 A/B, 한국어 대 영어 질문·설명. 다른 모델을 쓰지 않음.
- 추가 학습: 인코더 동결, 판단 헤드 soft cross-entropy, 2 epochs, micro_batch 4, grad_accum 4, head_lr 1e-4, choice 옵션 순서 shuffle, seed 42, CPU fp32. checkpoint selection이나 test 기반 hyperparameter 조정 없음. 공식 `train_model` 구현 사용. act head는 해당 loss의 직접 학습 대상이 아님.
- 학습과 추론 max_len 1,024 / head_max_len 256. 장문 진단만 명시적 override.
- 추가 학습 모델은 T=1 결과와 별도 calibration 80개에서 얻은 T 결과를 분리한다.
- 생성 sampling: 해당 없음. 옵션 점수 argmax, 자유 문장 생성 없음. GPU 없음. Torch intra-op 4 / inter-op 1 threads. CPU 모델명·메모리·라이브러리 버전 기록.
- 기준선: 균형 데이터 50% 무작위 및 50% 다수 클래스; 같은 학습 200개를 사용한 문자 2–5gram TF-IDF + LogisticRegression(C=1, max_iter=1000, random_state=42). BERT 분류기 및 Jev는 이 실험에서 측정하지 않음.

## 측정

정확도, macro F1, Wilson 95% 구간, 10개 equal-width bin top-label ECE, multiclass Brier(sum over classes), confidence >= .9 coverage/accuracy, 질문당 p50/p95 wall latency를 기록한다. 지연은 다운로드·초기 로드·별도 워밍업 3회를 제외하고 tokenizer부터 응답 생성까지 포함한다. 단일 프로세스 RSS와 가능하면 OS peak working set을 남긴다. 반복 평균 대신 해당 파일럿의 단일 순차 실행 분포다.

한국어 생성 및 출력 자연스러움은 지원하지 않는 기능이므로 N/A. 한국어 이해·입력 자연스러움·번역투 민감도는 분류 진단에서만 관찰한다. 한국어/영어 의미쌍의 tokenizer 토큰 수를 측정하되 6개 자작 쌍을 언어 전체의 토큰 효율로 일반화하지 않는다.

## 제한

환경 변경 기록: 사용자의 Docker Compose 지정에 따라 Windows 네이티브 사전 실행은 중단하고 `evaluation/local-preliminary/`로 격리했다. 최종 실험은 동일 데이터·프롬프트·학습 설정으로 Linux Docker 컨테이너에서 처음부터 수행한다. Compose CPU quota 4, memory limit 8 GiB, Torch 4 threads, 네트워크 차단. 기존 예비 결과를 보고 데이터·프롬프트·학습률을 조정하지 않았다. 최종 기사에는 Docker 결과만 실측으로 싣는다. 긴 filler를 만드는 tokenizer 호출에 명시적 max_length=1200을 추가했으며, 기존 encode 후 [:1200]과 같은 토큰 예산이다.

NSMC 감성 분류는 금융·법률·고객센터 성능을 대표하지 않는다. train/calibration/test 표본 규모가 작다. 기존 사전학습 데이터와 NSMC의 중복 여부는 확인 불가. 이번 학습은 인코더 전체 학습이나 RLCD의 효과를 평가하지 않는다. 진단 세트는 자동 작성된 소규모 편의 표본이며 신뢰구간을 붙여 일반 성능처럼 포장하지 않는다. 사용자 데이터는 외부 API에 전송하지 않는다.
