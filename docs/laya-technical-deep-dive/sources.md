# Laya Technical Deep Dive 자료집

조사 기준일 2026-10-05. 공식 구현은 커밋 `8a6e1328cce2460a0e5aa348ad465bb1b5821cd2`, 대상 HF 모델 revision은 `1720e3e3357cfe1e281542e223f8273b0890ca34`. `research/source-manifest.json`에 수집 URL·시각·해시를 보존했다. 아래 링크는 출처이고, 별도 표시가 없는 외부 수치를 직접 재현했다는 뜻은 아니다.

## 모델과 구현

| 자료 | 사용할 근거 | 해석할 때의 제한 |
| --- | --- | --- |
| [Laya Multilingual 모델 카드](https://huggingface.co/convaiinnovations/laya-multilingual) | 백본·크기·한국어 MASSIVE·언어별 한계 | 제작사 측정이며 100+ 언어가 동일 품질이라는 뜻 아님 |
| [Laya 저장소](https://github.com/NandhaKishorM/laya) | 모델 계열, API와 운영 도구 | main은 변함. 코드 해석은 보존 커밋 기준 |
| [common.py](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya/common.py) | 입력 토큰 구성, marker scorer, 추가 head, reward | 구현 사실과 설계 의도에 대한 해석 구분 |
| [agent.py](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya/agent.py) | 배치, 출력 schema, confidence 의미 | confidence/answer_confidence/act_probability 구분 |
| [train.py](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya/train.py) | RLCD/soft CE, encoder freeze, 옵션 shuffle, export | 편의 함수 분할이 문서가 아니라 질문 단위 |
| [학습 문서](https://nandhakishorm.github.io/laya/finetune/) | Kaggle·로컬 학습과 보정 절차 | typed-decisions 결과는 해당 분포의 추가 학습 실험 |
| [벤치마크 문서](https://nandhakishorm.github.io/laya/benchmarks/) | 조건별 결과와 알려진 한계 | 서로 다른 표본·장비의 수치를 합치지 않음 |
| [GLiClass](https://github.com/Knowledgator/GLiClass) | 동적 레이블 분류의 유사 접근 | 이 글에서 한국어 직접 대결은 미측정 |

## Jev와 독립 평가

앞부분의 등장 배경은 [TypeSafe 소개 문서](https://docs.typesafe.ai/introduction), [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer), [LAYA 개발자의 경위 설명](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)을 추가로 확인했다. Jev의 공식 발표일과 개발자의 반응은 구분해 기술했다. 유행의 규모나 과학적 최초성을 검증한 자료는 아니다. 개발자 글은 초기 ModernBERT-large 계열 설명이며 이 글의 Multilingual 모델·현재 학습 구현과 그대로 동일시하지 않는다.

개발자가 언급한 선행 논문은 [SalesRLAgent](https://arxiv.org/abs/2503.23303), [Confidence-Aware Routing](https://arxiv.org/abs/2510.01237)이다. 각각 판매 전환 확률 예측과 생성 전 신뢰도 기반 라우팅을 다룬다. 현재 LAYA Multilingual 구조를 직접 설명하는 논문으로 분류하지 않는다. 추가 원문·해시는 `research/source-manifest.json`에 기록했다.

| 자료 | 활용 |
| --- | --- |
| [TypeSafe 공식 발표](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | API의 성격·지원 후보 수·제작사 주장. 비공개 아키텍처는 추정하지 않음 |
| [sysone-bench v2 보고서](https://github.com/instax-dutta/sysone-bench/blob/master/results/v2/report-20260926/REPORT.md) | 동일 입력 정확도 비교. Laya 영어 모델 결과임을 명시 |
| [sysone-bench 정답 제작 조건](https://github.com/instax-dutta/sysone-bench#how-the-ground-truth-was-made) | AI 초안+1인 검토, 다국어 원어민 검증 없음 |
| [score 위치 편향 #131](https://github.com/NandhaKishorM/laya/issues/131) | 질문 순서에 민감한 사용자 재현 사례 |
| [레이블 표현 문제 #156](https://github.com/NandhaKishorM/laya/issues/156) | noul과 choice 레이블 표현에 따른 차이 |
| [RLCD ablation #741](https://github.com/NandhaKishorM/laya/issues/741) | soft CE와 RLCD의 비교 필요. 한 사용자의 제한된 실험 |
| [방글라어 추가 학습 사례](https://huggingface.co/nafiullah/laya-multilingual-bn-ecom-voice/blob/main/results/FINETUNE_RESULTS.md) | 개선·퇴행 동시 관찰. 일부 미검수 정답과 성공 기준 미통과를 함께 소개 |

## 논문

1. [BERT](https://arxiv.org/abs/1810.04805), Devlin et al., 2018 / NAACL 2019. 기반 인코더 설명.
2. [ModernBERT](https://arxiv.org/abs/2412.13663), Warner et al., 2024 / ACL 2025. 긴 문맥과 효율.
3. [mmBERT](https://arxiv.org/abs/2509.06888), Marone et al., 2025. 다국어 인코더, 언어 배합. Laya 성능과 동일시하지 않음.
4. [Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161), Yin et al., 2019. NLI 기반 접근과 평가 설계.
5. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599), Guo et al., ICML 2017. Temperature scaling.
6. [Overcoming Catastrophic Forgetting](https://arxiv.org/abs/1612.00796), Kirkpatrick et al., 2016 / PNAS 2017. 기존 기능 보존의 문제.
7. [Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647), Deußer et al., 2026 preprint. Jev 평가이며 Laya 직접 비교가 아님.
8. [Just Ask Jev](https://arxiv.org/abs/2609.29429), 2026 preprint. 질문과 입력 구성에 따른 정렬 실패 탐지 평가.
9. [OpenJev-RLCD](https://arxiv.org/abs/2609.38850), Gao and Wang, 2026 preprint. 별도 연구 구현이며 Jev의 공식 내부 공개가 아님.

동명의 EEG Laya 논문은 대상 모델과 무관해 근거에서 제외했다. 공식 Laya 대표 학술 논문은 이번 조사 범위에서 확인하지 못했다.

## 한국어 실측 자료

- [NSMC 공식 자료](https://github.com/e9t/nsmc): 영화 리뷰 감성 분류, CC0, 150k train / 50k test. 중립 평점은 배제된 데이터. 이 글에서는 균형 부분 표본만 사용.
- `evaluation/data/provenance.json`: 원본 NSMC revision과 해시, 선택 표본 해시.
- `research/evaluation-protocol.md`: 출력 확인 전 가설과 조건, Docker 전환 기록.
- `research/run_evaluation.py`: baseline, 추론, 추가 학습, 보정, 진단 측정 코드.
- `evaluation/results/`: 최종 Docker 실행 결과만 보관.
- `evaluation/local-preliminary/`: 중단한 Windows 예비 실행. 최종 결과와 섞지 않음.

## 그림 출처와 재사용

| 파일 | 종류 | 출처와 권리 표시 |
| --- | --- | --- |
| `01-dynamic-categories.webp` | 직접 작성한 설명 그림 | common.py의 동작을 해설. 예상 답과 실측 구분 |
| `02-architecture.webp` | 직접 작성한 아키텍처 | 공개 코드 커밋 기준, Jev 내부 구조 아님 |
| `03-finetuning-retention.webp` | 직접 작성한 운영 제안 | 사실 그림이 아니라 제안 구조 |
| `04-independent-benchmark.webp` | 공개 숫자 재시각화 | sysone-bench v2, 영어 Laya의 평가 조건 명시 |
| `05-official-long-context.webp` | 외부 원본 그림 변환 | Convai Innovations / NandhaKishorM/laya, Apache-2.0 저장소. `research/laya/LICENSE` 보존. 원본 PNG 보존, WebP 변환만 수행 |
| `06-korean-measurements.webp` | 직접 측정한 결과의 그래프 | Docker NSMC 시험 200건, Wilson 95% 구간. 외부 보고와 분리 |

직접 작성한 그림은 PNG와 편집 가능한 SVG를 함께 보관했다. 블로그에는 WebP만 삽입했다. 최적화 통계는 `assets/image-manifest.json`을 참고한다. 논문 그림을 출처 없이 재사용하지 않았다.
# 2026-10-07 민원 분류 후속 실험

- [laya-korean-eval 공개 저장소](https://github.com/sungreong/laya-korean-eval): 3×3×3 민원 taxonomy, 합성 학습·validation·시험 자료, Docker 실행 코드, LAYA와 KoBERT 문항별 예측 및 통계.
- [SKTBrain KoBERT 공식 저장소](https://github.com/SKTBrain/KoBERT): 12층, hidden 768, 최대 512 token, SentencePiece vocabulary 8,002 등 구조와 공개 학습 정보.
- [SK Telecom KoBERT 체크포인트](https://huggingface.co/skt/kobert-base-v1): 이번 실제 `[MASK]` adapter 실험에서 revision `359874884642d748079d4dd4ff547f2cdbff67d6`을 사용.
- [후속 실험 원 결과](https://github.com/sungreong/laya-korean-eval/tree/main/results/complaint-training-2026-10-07): LAYA 5 epoch, KoBERT 5 epoch, 81개 단일 정답·9개 정보 부족 시험의 예측과 paired 분석.

후속 민원 자료는 실제 상담 기록이나 공식 행정 분류표가 아니라 작성자가 만든 합성 자료다. 같은 작성자의 문장과 라벨 설명이 의미상 겹쳐 성능이 부풀려질 수 있다. KoBERT adapter는 공식 KoBERT 또는 LAYA 구현이 아니라 `[MASK]` hidden vector를 후보 점수로 사용할 수 있는지 확인한 실험 코드다.

