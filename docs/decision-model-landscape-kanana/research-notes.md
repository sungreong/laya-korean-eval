# 조사 메모

## 요청 계약

- 독자: AI/ML 엔지니어, 기술 리더, 제품 설계자
- 언어: 한국어
- 형식: 근거 중심 Technical Deep Dive 블로그
- 필수 내용: 기업 및 오픈 모델 타임라인, 아키텍처, 장단점, 가격·비용, 알려지지 않은 부분, 도입 판단
- 근거 우선순위: 공식 문서·모델 카드·GitHub·논문 → 독립 평가 → 보조 기사
- 최초 기준일: 2026-10-12, Kanana 보강일: 2026-11-08, Asia/Seoul

## 범위 정의

여기서 typed decision model은 다음 계약을 만족한다.

1. 입력은 공통 상태(state)와 하나 이상의 질문이다.
2. 선택지는 호출자가 런타임에 정의한다.
3. 출력은 선택지별 확률, 순서형 점수 또는 참/거짓 확률이다.
4. 자유 텍스트 생성이 핵심 출력 경로가 아니다.

따라서 일반적인 reranker, 고정 라벨 분류기, constrained JSON generation과 연관은 있지만 같은 범주로 취급하지 않았다.

## 비용 계산 가정

블로그의 월 비용 예시는 다음 가정으로 계산했다.

- 1건당 입력: 500 tokens
- 월간 처리량: 10,000,000건
- 총 입력: 5,000,000,000 tokens = 5,000 MTok
- 출력 토큰 비용: 각 공급사가 decision endpoint에서 무료라고 명시한 경우 0
- 네트워크, 로그, 저장, 재시도, human review, 최소 요금은 제외

공식은 `월 비용 = 입력 토큰 수 / 1,000,000 × 입력 단가`다.

## 증거 등급

- A: 공식 API 문서·공식 가격표·공개 모델 카드·공개 코드
- B: 저자 논문 또는 공급사 자체 benchmark
- C: 독립 benchmark·초기 survey
- D: 커뮤니티 모델 카드 또는 디렉터리 항목

모델 간 benchmark 수치는 같은 데이터·하드웨어·버전에서 측정되지 않았다면 직접 순위화하지 않았다.

## 링크된 영상 확인

- URL: https://youtube.com/shorts/EtWkF5Ocr8E
- 제목: `Jev 벌써 끝? MS Decision-1 미쳤다 #Shorts #결정모델 #Jev #마이크로소프트 #Microsoft #DecisionsAPI #Decision1`
- 채널: 잔재미코딩(DaveLee)
- 확인 방법: YouTube oEmbed metadata

## 알려진 조사 한계

- Jev의 파라미터 수와 내부 architecture는 공개되지 않았다.
- Microsoft-Decision-1의 세부 readout head와 학습 데이터는 공개되지 않았다.
- OpenAI Decisions API가 사용하는 세부 decision readout 구조는 공개되지 않았다.
- 공급사 benchmark는 평가셋, 네트워크 포함 여부, 하드웨어, batch 조건이 다르다.
- 최초 조사 시점에는 한국어 단독 benchmark를 공개한 주요 공급사를 확인하지 못했다. 2026-11-06 Kakao가 Kanana Decision Bench 72.50을 공개했지만 15,909문항 데이터는 비공개다.
- DecisionEval의 140개 디렉터리는 생태계 규모를 보여주지만 각 모델의 품질을 보증하지 않는다.

## Kanana 로컬 실험 계약

- model revision: `1489062def4145ae3a0a9261fbac4f4d88a8c6d1`
- weight SHA-256: `528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b`
- runtime: Docker Desktop WSL2, PyTorch 2.10.0 CPU, Transformers 5.5.3, BF16, SDPA
- hardware quota: Ryzen 7 PRO 7840U, CPU 8 thread, memory 11GiB
- fixture: 기존 합성 민원 90건, 단일 정답 81건, candidate 27개
- task-specific fine-tuning: 없음
- scope: 저자 작성 합성 진단이며 독립 검수·production benchmark가 아님

## Kanana GPU·서빙·학습·라이선스 보강

- Kakao 공식 자료는 H200 GPU 1장, 한국어 상품 리뷰의 판단 1건 약 42ms를 보고하지만 batch, 후보 수와 cache 상태가 로컬 실험과 같지 않아 CPU 6.65초와 직접 속도 배수로 비교하지 않는다.
- 로컬 공개 구현의 candidate cache는 2,560차원 FP32 vector를 최대 4,096개 저장한다. 순수 vector 크기는 27개 0.26MiB, 300개 2.93MiB, 4,096개 40MiB다.
- reference server는 HTTP thread와 queue를 제공하지만 단일 `Decider.turn` lock으로 model inference를 직렬화한다. Production 설계는 별도의 queue, worker 수, warm-up, 인증과 관측성을 검증해야 한다.
- 공식 checkpoint와 inference 코드는 fine-tuning 가능성을 열지만 training script와 recipe는 없다. `embed_messages`가 `no_grad`이므로 custom trainable forward가 필요하다.
- Kanana Open License는 자사 서비스 운영을 허용하지만 유료 API/cloud access, 제3자 SI·on-prem, on-device 판매에는 Kakao의 별도 commercial license를 요구한다. 재배포·파생 모델에는 Notice와 `Powered by Kanana` 등 조건이 있다.
- 사용자 제공 Kakao Tech 영상 `2crSurA7R-U`의 제목과 설명을 확인했다. 영상 대상은 Kanana-v-embedding이며 V-Decide 자체가 아니므로, 멀티모달 embedding 학습·검색 서비스 적용의 계열 배경으로만 사용하고 V-Decide fine-tuning 근거로 일반화하지 않는다.
