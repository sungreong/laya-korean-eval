# 데이터 확대·epoch 실험 사전 계획 (2026-10-07)

목적: 한국어 실제 학습 표본과 학습 반복을 늘렸을 때 성능이 개선되는지 확인한다. 기존 200개·2 epoch 결과를 역사적 비교로 사용하되 데이터 크기와 총 optimizer update 수를 분리한 인과 실험이라고 해석하지 않는다.

- 고정된 기존 NSMC 원본 사용. 합성 리뷰/가짜 정답을 추가하지 않는다.
- 학습 1,000개: 기존 학습 200개 + 공식 train에서 새로 뽑은 800개, 긍정/부정 균형.
- 검증 200개: 공식 train에서 별도 추출, epoch 선택 전용. 기존 calibration 80개는 학습/검증에서 제외한다.
- 시험: 기존 test 200개는 비교용, 공식 test의 별도 500개는 추가 확인용. 모든 분할의 정확히 같은 본문을 제거하며, test 전체의 텍스트는 train/validation 후보에서 제외한다. 의미상 중복까지 검사하지 않는다.
- seed 42, 한국어 질문과 A/B 선택지 유지. max_len 1024, head_max_len 256.
- 5 epochs, 인코더 동결, soft-CE, head_lr 1e-4, micro_batch 4, grad_accum 4, cosine scheduler, CPU FP32. 옵션 순서 shuffle. 총 학습 horizon은 5로 고정하며 중간 epoch는 독립적인 1/2 epoch 학습과 같지 않다.
- 매 epoch validation accuracy, 동률이면 NLL로 선택한다. test는 epoch 선택에 사용하지 않는다. optimizer/scheduler를 epoch 사이에 재시작하지 않는다.
- 선택한 head를 복원하여 upstream FP16 export 후 다시 로드한다. 최종 시험 점수는 실제 저장 모델의 결과다. 온도 보정은 하지 않는다.
- 기본 모델과 저장된 학습 모델을 기존 test 200개 및 새 test 500개, 부서 분류 진단 24개에서 비교한다. 새로운 500개는 이번 실행에서 처음 평가하지만 단일 seed의 소규모 후속 실험이며 확증적 성능 주장에 사용하지 않는다.
- 같은 확대 학습 표본을 사용하는 TF-IDF+LR 기준선도 평가한다.
- 정확도, Macro F1, Wilson 95% 구간, ECE/Brier, paired bootstrap 차이 구간과 탐색적 McNemar p 값을 기록한다. 다중 비교 보정 없음. 생성 성능은 해당 없음.
- 학습 손실이 내려가도 검증·시험 성능이 좋아지지 않으면 개선으로 보고하지 않는다. 전체 인코더 학습이나 다중 seed 결과로 일반화하지 않는다.
