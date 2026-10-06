from pathlib import Path
import csv, io, requests, random, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evaluation/data'; DATA.mkdir(parents=True,exist_ok=True)
rng=random.Random(42)
def write(name, rows):
    p=DATA/name;p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    return hashlib.sha256(p.read_bytes()).hexdigest()
ko={'sentiment':{'type':'choice','instructions':'영화 리뷰 전체의 감정이 긍정인지 부정인지 고르세요.','criteria':{'A':'긍정적인 영화 리뷰: 영화에 만족하거나 추천한다.','B':'부정적인 영화 리뷰: 영화에 불만족하거나 추천하지 않는다.'}}}
en={'sentiment':{'type':'choice','instructions':'Classify the overall sentiment of this movie review as positive or negative.','criteria':{'A':'A positive movie review: the reviewer is satisfied or recommends the film.','B':'A negative movie review: the reviewer is dissatisfied or does not recommend the film.'}}}
questions={'ko':ko,'en':en,'department':{'department':{'type':'choice','instructions':'고객이 지금 해결하고 싶은 핵심 문제를 기준으로 담당 부서를 하나 고르세요. 이전 문제보다 현재 요청을 우선하세요.','criteria':{'billing':'중복 결제, 청구 금액, 환불, 결제 승인 취소','technical':'서비스 장애, 프로그램 오류, 기능이 작동하지 않음','delivery':'상품 배송, 택배 위치, 배송 지연','sales':'구매 전 견적, 가격, 할인, 도입 상담','account':'비밀번호 변경, 계정 탈퇴, 계정 정보 수정','other':'위 부서가 담당하지 않는 요청 또는 단순 인사'}}}}
(DATA/'questions.json').write_text(json.dumps(questions,ensure_ascii=False,indent=2),encoding='utf-8')
source=[]; seen=set()
commit='cc0670e872d4ac27bfe36c87456783004b39ef6c'
all_rows={}
for split in ['train','test']:
    url=f'https://raw.githubusercontent.com/e9t/nsmc/{commit}/ratings_{split}.txt'
    res=requests.get(url,timeout=60);res.raise_for_status();(DATA/f'ratings_{split}.txt').write_bytes(res.content)
    source.append({'url':url,'sha256':hashlib.sha256(res.content).hexdigest(),'bytes':len(res.content)})
    rows=[]
    for r in csv.DictReader(io.StringIO(res.content.decode('utf-8')),delimiter='\t'):
        t=r['document'].strip()
        if not t or t in seen: continue
        seen.add(t); rows.append({'id':r['id'],'text':t,'label':int(r['label'])})
    all_rows[split]=rows
def balanced(rows,n):
    samples=[]
    for lab in [0,1]:samples+=rng.sample([r for r in rows if r['label']==lab],n//2)
    rng.shuffle(samples);return samples
selected=balanced(all_rows['train'],280)
train=[];cal=[]
for lab in [0,1]:
    group=[r for r in selected if r['label']==lab];train+=group[:100];cal+=group[100:]
rng.shuffle(train);rng.shuffle(cal)
test=balanced(all_rows['test'],200)
def public_rows(rows):
    return [{'id':r['id'],'state':r['text'],'questions':ko,'gold':{'sentiment':{'probabilities':{'A':float(r['label']==1),'B':float(r['label']==0)}}}} for r in rows]
digests={}
for name,rows in [('train.jsonl',train),('calibration.jsonl',cal),('test.jsonl',test)]:digests[name]=write(name,public_rows(rows))

diag=[
('natural','요금이 두 번 빠져나갔어요. 하나는 돌려주세요.','billing'),
('natural','앱에서 확인 버튼을 눌러도 아무 반응이 없네요.','technical'),
('natural','지난주 주문한 물건이 아직 안 왔어요. 어디쯤인가요?','delivery'),
('natural','우리 팀 스무 명이 쓸 건데 얼마인지 알려주세요.','sales'),
('negation','환불을 원하지는 않아요. 비밀번호를 바꾸고 싶어요.','account'),
('negation','배송이 늦은 게 아닙니다. 돈이 중복해서 나갔어요.','billing'),
('negation','가격을 물어보는 게 아니라 결제 후 화면이 멈추는 문제예요.','technical'),
('negation','물건을 취소하려는 건 아니에요. 택배 위치만 확인해 주세요.','delivery'),
('context','이전 상담: 환불을 문의함. 고객의 현재 요청: 환불은 해결됐습니다. 이제 계정을 삭제해 주세요.','account'),
('context','이전 상담: 앱 오류 접수. 고객의 현재 요청: 오류는 고쳤어요. 회사 도입 견적을 받고 싶습니다.','sales'),
('context','상담원: 배송 문의인가요? 고객: 아니요, 상품은 받았는데 청구 금액이 주문 금액보다 큽니다.','billing'),
('context','상담원: 무엇을 도와드릴까요? 고객: 따로 요청은 없고 오늘 친절한 상담 감사드립니다.','other'),
('mixed','API request가 계속 timeout 납니다. endpoint 장애를 확인해 주세요.','technical'),
('mixed','tracking number는 있는데 parcel 위치가 며칠째 그대로예요.','delivery'),
('mixed','Enterprise plan 50 seats 기준 quotation 부탁드립니다.','sales'),
('mixed','password를 변경하고 profile에 등록한 이름도 수정하고 싶어요.','account'),
('terminology','매입 확정 전 승인 취소한 거래가 청구서에 반영됐습니다. 정정해 주세요.','billing'),
('terminology','웹훅 수신 서버가 502를 반환하고 재시도 큐가 누적됩니다.','technical'),
('terminology','운송장 조회에 간선하차 이후 이동 이력이 없습니다. 배송 상태 확인 부탁드립니다.','delivery'),
('terminology','계약 전 대량 구매 단가와 연간 약정 할인율을 제안해 주세요.','sales'),
('translationese','나는 나의 계정으로부터 탈퇴를 수행하는 것을 원합니다.','account'),
('translationese','나는 당신에게 좋은 아침을 기원하는 중입니다. 별도의 요청은 없습니다.','other'),
('translationese','당신의 친절에 대한 감사를 표현하고 싶습니다. 추가 지원은 필요하지 않습니다.','other'),
('translationese','이 메시지의 목적은 인사를 전달하는 것입니다. 처리 요청은 존재하지 않습니다.','other'),
]
digests['diagnostics.jsonl']=write('diagnostics.jsonl',[{'id':f'd{i:02}','slice':sl,'state':txt,'expected':lab} for i,(sl,txt,lab) in enumerate(diag)])
pairs=[('결제가 두 번 됐어요.','I was charged twice.'),('배송이 아직 도착하지 않았습니다.','The delivery has not arrived yet.'),('비밀번호를 변경하고 싶어요.','I want to change my password.'),('이 영화는 정말 재미있었어요.','This movie was really entertaining.'),('오류는 해결됐지만 환불은 아직입니다.','The error is fixed but the refund is still pending.'),('웹훅 수신 서버가 응답하지 않습니다.','The webhook receiving server is not responding.')]
(DATA/'token_pairs.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2),encoding='utf-8')
(DATA/'provenance.json').write_text(json.dumps({'seed':42,'nsmc_commit':commit,'sources':source,'selected_sha256':digests,'sizes':{'train':len(train),'calibration':len(cal),'test':len(test),'diagnostic':len(diag)}},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'commit':commit,'sizes':[len(train),len(cal),len(test),len(diag)]}))
