"""Korean choice example; no claimed output until actually executed."""
import json
import laya
from pathlib import Path

local=Path(__file__).resolve().parents[1]/'evaluation/models/base'
agent=laya.load(str(local),device='cpu')
state={'body':'결제가 두 번 됐어요. 중복된 건만 환불해 주세요.'}
questions={'department':{'type':'choice','instructions':'고객 문의를 처리할 담당 부서를 고르세요.','criteria':{'billing':'결제, 청구, 중복 결제, 환불','technical':'오류, 장애, 로그인 문제','sales':'가격, 견적, 구매 전 상담','other':'위 부서에 해당하지 않는 문의'}}}
print(json.dumps(agent.predict(state,questions),ensure_ascii=False,indent=2))
