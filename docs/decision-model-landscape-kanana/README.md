# Decision Model Landscape 2026

2026년 10월 12일 최초 조사 뒤, 11월 8일 Kakao Kanana-2 V-Decide의 공식 자료와 동일 민원 CPU 실험을 보강한 블로그 제작 폴더입니다.

## 파일

- `blog.md`: 최종 한국어 기술 블로그 원고
- `blog.html`: Markdown Writer로 렌더링한 standalone 미리보기
- `blog-embed.html`: CMS 삽입용 렌더
- `sources.md`: 주장별 1차 출처와 보조 출처
- `research-notes.md`: 조사 범위, 용어, 비용 계산 가정, 한계
- `data/model-landscape.csv`: 주요 모델 비교 데이터
- `assets/common-architecture.svg`: 공통 추론 아키텍처
- `assets/timeline.svg`: 2026년 9~11월 공개 타임라인
- `assets/kanana2-v-decide-architecture-benchmark-ko.webp`: Kanana 아키텍처·동일 데이터 성능 비교 인포그래픽

## Kanana CPU 실험 연결

- 코드와 Docker Compose: 저장소 루트 `../../`
- 결과: `../../results/kanana-2-v-decide-2026-11-08/`
- 조건: BF16, CPU 8 thread, memory 11GiB, 27개 후보, 단일 정답 81건
- 결과: 75/81, accuracy 92.6%, Macro-F1 92.1%, warm p50 6.65초

## 조사 범위

이 문서에서 decision model은 **사용자가 런타임에 정의한 유한 선택지에 대해 자유 텍스트를 생성하지 않고 확률 분포를 반환하는 모델 또는 전용 API**를 뜻합니다. Decision Transformer, 의사결정나무, 일반적인 LLM JSON 출력은 별도 계열로 보았습니다.

전체 Hugging Face 모델을 무차별 나열하지 않고 다음 기준 중 하나를 만족하는 대표 제품과 공개 구현을 다뤘습니다.

1. 기업이 공식 발표한 상용 API
2. 기업·연구팀이 아키텍처와 가중치를 공개한 모델
3. 재현 가능한 모델 카드·코드·논문이 있는 주요 오픈 구현

보강 기준일 뒤 가격과 모델 버전은 바뀔 수 있습니다. 실제 도입 전 공식 가격표, 모델 카드와 Kanana Open License를 다시 확인해야 합니다.
