from pathlib import Path
import os,sys,json,time,math,platform,hashlib,random,gc
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/laya'))
os.environ['USE_TF']='0';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
import torch, transformers, psutil
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import laya
from laya.train import TrainConfig,items_from_rows,train_model,calibration_records,save_checkpoint
from laya.calibrate import fit_temperature_map
torch.set_num_threads(4);torch.set_num_interop_threads(1)
torch.manual_seed(42);random.seed(42);np.random.seed(42)
DATA=ROOT/'evaluation/data';OUT=ROOT/os.environ.get('LAYA_RESULTS_DIR','evaluation/results')
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'summary.json').exists():
    raise RuntimeError('Existing results are preserved. Set LAYA_RESULTS_DIR to a fresh relative directory.')
Q=json.loads((DATA/'questions.json').read_text('utf-8'))
def read(name):return [json.loads(x) for x in (DATA/name).read_text('utf-8').splitlines() if x.strip()]
train,cal,test,diag=[read(x) for x in ['train.jsonl','calibration.jsonl','test.jsonl','diagnostics.jsonl']]
def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def memory():
    m=psutil.Process().memory_info();result={k:getattr(m,k)/2**20 for k in ['rss','peak_wset'] if hasattr(m,k)}
    if sys.platform=='linux':
        import resource
        result['peak_rss']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    return result
def expected(r):return max(r['gold']['sentiment']['probabilities'],key=r['gold']['sentiment']['probabilities'].get)
def wilson(correct,n):
    p=correct/n;z=1.95996398454;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]
def metrics(rows):
    good=np.array([r['expected']==r['predicted'] for r in rows]); conf=np.array([r['confidence'] for r in rows]);n=len(rows)
    ece=0.
    for i in range(10):
        sel=(conf>=i/10)&((conf<(i+1)/10) if i<9 else (conf<=1))
        if sel.any():ece+=sel.mean()*abs(good[sel].mean()-conf[sel].mean())
    brier=np.mean([sum((p-float(k==r['expected']))**2 for k,p in r['probabilities'].items()) for r in rows])
    high=conf>=.9
    return {'n':n,'correct':int(good.sum()),'accuracy':float(good.mean()),'wilson95':wilson(int(good.sum()),n),'macro_f1':f1_score([r['expected'] for r in rows],[r['predicted'] for r in rows],average='macro'),'ece10_top_label':float(ece),'brier_sum':float(brier),'high_confidence_n':int(high.sum()),'coverage_at_0.9':float(high.mean()),'accuracy_at_0.9':float(good[high].mean()) if high.any() else None,'latency_ms_p50':float(np.percentile([r['latency_ms'] for r in rows],50)),'latency_ms_p95':float(np.percentile([r['latency_ms'] for r in rows],95))}

summary={'protocol':'research/evaluation-protocol.md','environment':{'python':sys.version,'platform':platform.platform(),'cpu':'AMD Ryzen 7 PRO 7840U','ram_bytes':psutil.virtual_memory().total,'torch':torch.__version__,'transformers':transformers.__version__,'laya':laya.__version__,'threads':4,'interop_threads':1,'device':'cpu','dtype':'float32','code_commit':'8a6e1328cce2460a0e5aa348ad465bb1b5821cd2','model_revision':'1720e3e3357cfe1e281542e223f8273b0890ca34'},'stages':{}}
for name,path in [('cpu_max','/sys/fs/cgroup/cpu.max'),('memory_max','/sys/fs/cgroup/memory.max')]:
    if Path(path).exists():summary['environment'][name]=Path(path).read_text().strip()
summary['execution']='Docker Compose' if Path('/.dockerenv').exists() else 'native preliminary'
summary['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
save('summary.json',summary)

# Baseline sees only the same 200 labelled training rows.
t0=time.perf_counter();vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=1)
X=vec.fit_transform([r['state'] for r in train]);clf=LogisticRegression(C=1,max_iter=1000,random_state=42)
clf.fit(X,[expected(r) for r in train]);baseline_train_s=time.perf_counter()-t0
for _ in range(3):clf.predict_proba(vec.transform([test[0]['state']]))
br=[]
for r in test:
    t=time.perf_counter();prob=clf.predict_proba(vec.transform([r['state']]))[0];ms=(time.perf_counter()-t)*1000
    pr=dict(zip(clf.classes_,map(float,prob)));pred=max(pr,key=pr.get)
    br.append({'id':r['id'],'expected':expected(r),'predicted':pred,'probabilities':pr,'confidence':pr[pred],'latency_ms':ms})
save('baseline_predictions.json',br);summary['stages']['tfidf_logreg']=dict(metrics(br),train_seconds=baseline_train_s,features=len(vec.vocabulary_));save('summary.json',summary)

t=time.perf_counter();agent=laya.load(str(ROOT/'evaluation/models/base'),device='cpu',compile=False)
summary['load_seconds']=time.perf_counter()-t
summary['parameter_count']=sum(p.numel() for p in agent.model.parameters())
summary['encoder_parameter_count']=sum(p.numel() for p in agent.model.encoder.parameters())
summary['parameter_bytes']=sum(p.numel()*p.element_size() for p in agent.model.parameters())
summary['memory_after_load_mib']=memory();save('summary.json',summary)
print('MODEL LOADED',summary['load_seconds'],summary['memory_after_load_mib'],flush=True)

def run_sentiment(name,questions):
    for _ in range(3):agent.predict(test[0]['state'],questions)
    rows=[]
    for i,r in enumerate(test):
        t=time.perf_counter();out=agent.predict(r['state'],questions);ms=(time.perf_counter()-t)*1000
        a=out['answers']['sentiment']
        rows.append({'id':r['id'],'expected':expected(r),'predicted':a['choice'],'probabilities':a['probabilities'],'confidence':a['answer_confidence'],'latency_ms':ms,'response':out})
        if (i+1)%50==0:print(name,i+1,flush=True)
    save(name+'_predictions.json',rows);summary['stages'][name]=metrics(rows);summary['stages'][name]['process_memory_mib']=memory();save('summary.json',summary)
    print(name,summary['stages'][name],flush=True);return rows
def run_diag(name):
    rows=[]
    for r in diag:
        t=time.perf_counter();out=agent.predict(r['state'],Q['department']);ms=(time.perf_counter()-t)*1000;a=out['answers']['department']
        rows.append(dict(r,predicted=a['choice'],probabilities=a['probabilities'],confidence=a['answer_confidence'],latency_ms=ms,response=out))
    save(name+'_predictions.json',rows)
    summary['stages'][name]={'all':metrics(rows),'by_slice':{s:metrics([r for r in rows if r['slice']==s]) for s in sorted({r['slice'] for r in rows})}}
    save('summary.json',summary);print(name,summary['stages'][name]['all'],flush=True);return rows

base=run_sentiment('base_ko',Q['ko']);base_en=run_sentiment('base_en',Q['en']);run_diag('base_diagnostic')
summary['prompt_pair']={'disagreement_n':sum(a['predicted']!=b['predicted'] for a,b in zip(base,base_en)),'ko_only_correct':sum(a['predicted']==a['expected'] and b['predicted']!=b['expected'] for a,b in zip(base,base_en)),'en_only_correct':sum(a['predicted']!=a['expected'] and b['predicted']==b['expected'] for a,b in zip(base,base_en))}
token_rows=[]
for ko,en in json.loads((DATA/'token_pairs.json').read_text('utf-8')):
    token_rows.append({'ko':ko,'en':en,'ko_tokens':len(agent.tok.encode(ko,add_special_tokens=False)),'en_tokens':len(agent.tok.encode(en,add_special_tokens=False))})
save('tokenization.json',token_rows)
summary['tokenization']={'ko_total':sum(r['ko_tokens'] for r in token_rows),'en_total':sum(r['en_tokens'] for r in token_rows),'pairs':len(token_rows)}

# Long context: exact tokenized unrelated filler, current request at front or back.
filler='기록 보관용 안내입니다. 아래 내용은 현재 문의와 무관한 일반 공지입니다. 오늘 사무실의 식물에 물을 주었고 회의실의 의자를 정리했습니다. '
fids=agent.tok.encode(filler*200,add_special_tokens=False,truncation=True,max_length=1200);filler=agent.tok.decode(fids)
request='현재 고객 요청: 결제가 두 번 승인되었습니다. 중복 결제된 금액을 환불해 주세요.'
longq={'department':{'type':'choice','instructions':'현재 고객 요청의 담당 부서를 고르세요. 기록 보관용 공지는 무시하세요.','criteria':Q['department']['department']['criteria']}}
longrows=[]
for position in ['front','back']:
    state=request+'\n'+filler if position=='front' else filler+'\n'+request
    for budget in [1024,2048]:
        t=time.perf_counter();out=agent.predict(state,longq,max_len=budget);ms=(time.perf_counter()-t)*1000
        longrows.append({'position':position,'max_len':budget,'state_tokens':len(agent.tok.encode(state,add_special_tokens=False)),'expected':'billing','latency_ms':ms,'response':out})
save('long_context.json',longrows);summary['memory_after_long_context_mib']=memory();save('summary.json',summary)

# Train only after all base measurements. Calibration and test never enter the optimizer.
ti,skip1=items_from_rows(agent.tok,train,1024,256);ci,skip2=items_from_rows(agent.tok,cal,1024,256)
assert not skip1 and not skip2
assert not ({r['state'] for r in train}&{r['state'] for r in cal})
assert not ({r['state'] for r in train+cal}&{r['state'] for r in test})
cfg=TrainConfig(epochs=2,micro_batch=4,grad_accum=4,head_lr=1e-4,loss='soft-ce',shuffle_options=('choice',),freeze_encoder=True,seed=42,amp=False,gradient_checkpointing=False,log_every=10,max_len=1024,head_max_len=256)
t=time.perf_counter();losses=train_model(agent.model,agent.tok,ti,cfg,torch.device('cpu'),1024,256)
summary['training']={'seconds':time.perf_counter()-t,'epoch_loss':losses,'rows':len(ti),'freeze_encoder':True,'epochs':2,'head_lr':1e-4,'loss':'soft-ce','memory_mib':memory()};save('summary.json',summary)
agent.temperature=[1.,1.,1.];agent.temperature_by_options={}
run_sentiment('finetuned_raw',Q['ko']);run_diag('finetuned_diagnostic_raw')
records=calibration_records(agent.model,agent.tok,ci,torch.device('cpu'),1024,256,batch_size=4)
fitted=fit_temperature_map(records);save('calibration.json',fitted)
agent.temperature=fitted['temperature'];agent.temperature_by_options=fitted['temperature_by_options']
summary['calibration']=fitted;save('summary.json',summary)
run_sentiment('finetuned_calibrated',Q['ko'])
out_cfg=dict(agent.cfg,temperature=fitted['temperature'],fine_tuned=True)
out_cfg.pop('temperature_by_options',None)
if fitted['temperature_by_options']:out_cfg['temperature_by_options']=fitted['temperature_by_options']
save_checkpoint(agent.model,agent.tok,out_cfg,str(ROOT/'evaluation/models/finetuned'))
summary['saved_checkpoint']='evaluation/models/finetuned'
summary['note']='Reported fine-tuned inference used live fp32 weights; exported weights are fp16 per upstream save_checkpoint and are not independently rebenchmarked.'
save('summary.json',summary)
print('FINISHED',flush=True)
