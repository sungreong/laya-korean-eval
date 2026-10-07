# LAYA Multilingual Technical Deep Dive

한국어 기술 블로그의 배포용 묶음입니다.

- `blog.html`: 브라우저에서 바로 여는 standalone 문서. 본문 WebP 이미지 8개가 Base64로 내장되어 있습니다.
- `blog-embed.html`: 블로그/CMS 삽입용 HTML.
- `blog.md`: 수정 가능한 원고.
- `sources.md`: 논문, 공식 문서, 모델 카드와 후속 실험 출처.
- `assets/`: 그림 원본과 최적화 WebP. HTML은 최적화본만 내장합니다.
- `evaluation/`, `research/`: 본문에서 직접 연결한 최초 한국어 파일럿의 최소 근거 파일.

후속 민원 실험의 실행 코드는 이 저장소 루트의 `research/`, 데이터는 `datasets/complaints/`, 결과는 `results/complaint-training-2026-10-07/`와 `results/hierarchical-prefix-2026-10-07/`에 있습니다. 블로그 13절에 3×3×3 taxonomy, 명확·간접·복합·정보 부족 사례, LAYA 추가 학습, 실제 `[MASK]` KoBERT 비교, 대·중·소 단계별 학습과 predicted-prefix 평가를 정리했습니다. 최종 체크포인트로 기존 독립 시험 90건과 미등록 유형 27건을 세 추론 방식에서 전부 다시 예측했으며 이전 예측값을 재사용하지 않았습니다. 최종 독립 시험에서 LAYA 일괄 방식은 34.6%, 순차+prefix는 23.5%, KoBERT `[MASK]` 순차+prefix는 16.0%였습니다. 학습에서 제외한 9개 유형도 추론 때 후보로 추가했지만 정확도는 불안정했습니다. 작은 합성 시험과 제한된 CPU·단일 seed·동결 encoder 조건이므로 모델의 우열을 결론 내리지 않습니다.

```bash
git clone https://github.com/sungreong/laya-korean-eval.git
cd laya-korean-eval
start docs\laya-technical-deep-dive\blog.html
```

Linux/macOS에서는 `open` 또는 사용 중인 브라우저로 `blog.html`을 여세요.

## 렌더 검증

- standalone 이미지 8개 모두 WebP Base64 내장 및 로드 성공
- 1440px와 390px에서 문서 폭 일치, 가로 overflow 0건
- JavaScript page error 0건
- frontmatter: `toc: false`, `appearance-radius: none`, `appearance-frame: none`, `viewer-chrome: hidden`

계층·prefix 결과 그림은 원본 PNG 95,487 bytes와 WebP quality 85 최적화본 49,442 bytes를 함께 보존했습니다. 크기는 2384×897이며 48.22% 감소했습니다. 전체 이미지 원본 711,374 bytes는 최적화 후 351,216 bytes가 됐습니다.
