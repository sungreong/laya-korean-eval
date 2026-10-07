# LAYA Multilingual Technical Deep Dive

한국어 기술 블로그의 배포용 묶음입니다.

- `blog.html`: 브라우저에서 바로 여는 standalone 문서. 본문 WebP 이미지 7개가 Base64로 내장되어 있습니다.
- `blog-embed.html`: 블로그/CMS 삽입용 HTML.
- `blog.md`: 수정 가능한 원고.
- `sources.md`: 논문, 공식 문서, 모델 카드와 후속 실험 출처.
- `assets/`: 그림 원본과 최적화 WebP. HTML은 최적화본만 내장합니다.
- `evaluation/`, `research/`: 본문에서 직접 연결한 최초 한국어 파일럿의 최소 근거 파일.

후속 민원 실험의 실행 코드는 이 저장소 루트의 `research/`, 데이터는 `datasets/complaints/`, 결과는 `results/complaint-training-2026-10-07/`에 있습니다. 블로그 13절에 3×3×3 taxonomy, 명확·간접·복합·정보 부족 사례, LAYA 추가 학습, 실제 `[MASK]` KoBERT 비교와 전체 Docker 명령을 정리했습니다.

```bash
git clone https://github.com/sungreong/laya-korean-eval.git
cd laya-korean-eval
start docs\laya-technical-deep-dive\blog.html
```

Linux/macOS에서는 `open` 또는 사용 중인 브라우저로 `blog.html`을 여세요.

## 렌더 검증

- standalone 이미지 7개 모두 WebP Base64 내장 및 로드 성공
- 1440px와 390px에서 문서 폭 일치, 가로 overflow 0건
- JavaScript page error 0건
- frontmatter: `toc: false`, `appearance-radius: none`, `appearance-frame: none`, `viewer-chrome: hidden`

새로 추가한 학습 곡선은 원본 PNG 78,981 bytes와 WebP quality 85 최적화본 40,744 bytes를 함께 보존했습니다. 크기는 2096×666이며 48.41% 감소했습니다.
