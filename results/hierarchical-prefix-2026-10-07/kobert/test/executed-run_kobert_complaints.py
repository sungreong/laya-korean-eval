"""Evaluate the trained KoBERT real-[MASK] shared scorer on held-out complaints."""
from pathlib import Path
import argparse, hashlib, json, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research'))
import torch
from complaint_schema import build_questions,question
from kobert_mask_model import collate,load_trained,pack_one
from run_complaints import summarize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fixture',default='datasets/complaints');args=ap.parse_args()
    out=ROOT/args.out
    if out.exists(): raise RuntimeError('Choose a fresh output directory')
    out.mkdir(parents=True);fixture=ROOT/args.fixture;taxonomy=json.loads((fixture/'taxonomy.json').read_text('utf-8'))
    cases=json.loads((fixture/'cases.json').read_text('utf-8'));leaves,_,q=build_questions(taxonomy)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);model,config=load_trained(ROOT/'evaluation/models/kobert-base-v1',ROOT/args.checkpoint)
    model.eval();tokenizer=model.tokenizer
    summary={'status':'running','model':'KoBERT real-[MASK] shared scorer','checkpoint':args.checkpoint,'fixture':args.fixture,'checkpoint_config':config,
             'fixture_sha256':{n:hashlib.sha256((fixture/n).read_bytes()).hexdigest() for n in ['taxonomy.json','cases.json']},
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'methods':{}}
    (out/'executed-run_kobert_complaints.py').write_bytes(Path(__file__).read_bytes())
    def save(name,value):(out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    save('summary.json',summary)

    def ask(state,criteria,stage):
        packed=pack_one(tokenizer,state,question(criteria)['route']['instructions'],criteria,max_length=512)
        batch=collate([packed],tokenizer.pad_token_id)
        with torch.no_grad(): logits=model(**batch);values=torch.softmax(logits[0],-1).cpu().tolist()
        choice=packed['keys'][max(range(len(values)),key=values.__getitem__)]
        return choice,{'stage':stage,'candidate_count':len(criteria),'probabilities':dict(zip(packed['keys'],values)),
                       'mask_token_id':tokenizer.mask_token_id,'marker_token_ids':[packed['ids'][p] for p in packed['markers']],
                       'tokens':len(packed['ids']),'state_tokens_dropped':packed['state_tokens_dropped'],
                       'option_tokens_dropped':packed['option_tokens_dropped'],
                       'option_token_budget':packed['option_token_budget'],
                       'usage':{'truncated':packed['state_tokens_dropped']>0 or packed['option_tokens_dropped']>0}}

    def predict(case,method):
        state=f"상담 경로: {case['channel']}\n민원 요약: {case['summary']}";tick=time.perf_counter()
        if method=='flat27': leaf,t=ask(state,q['flat'],'leaf27');trace=[t]
        elif method=='cascade':
            major,t1=ask(state,q['top'],'major3');middle,t2=ask(state,q['middle'][major],'middle3');leaf,t3=ask(state,q['bottom'][middle],'leaf3');trace=[t1,t2,t3]
        else:
            major,t1=ask(state,q['top'],'major3');major_name=next(x['name'] for x in taxonomy['majors'] if x['id']==major)
            middle_state=state+f"\n이전 단계 선택 대분류: {major_name}"
            middle,t2=ask(middle_state,q['middle'][major],'middle3')
            major_node=next(x for x in taxonomy['majors'] if x['id']==major)
            middle_name=next(x['name'] for x in major_node['children'] if x['id']==middle)
            leaf_state=middle_state+f"\n이전 단계 선택 중분류: {middle_name}"
            leaf,t3=ask(leaf_state,q['bottom'][middle],'leaf3');trace=[t1,t2,t3]
        return dict(case,predicted_leaf=leaf,predicted_path=leaves[leaf]['path'],
                    expected_path=leaves[case['expected_leaf']]['path'] if case['expected_leaf'] else None,
                    acceptable_paths=[leaves[k]['path'] for k in case['acceptable_leaves']],
                    latency_ms=(time.perf_counter()-tick)*1000,trace=trace)

    for method in ['flat27','cascade','cascade_prefix']:
        predict(cases[0],method);rows=[]
        for i,case in enumerate(cases,1):
            rows.append(predict(case,method))
            if i%10==0: print(method,i,'/',len(cases),flush=True)
        if not all(all(x==tokenizer.mask_token_id for x in t['marker_token_ids']) for r in rows for t in r['trace']):
            raise RuntimeError('A marker position is not the real tokenizer [MASK] id')
        save(method+'_predictions.json',rows);summary['methods'][method]=summarize(rows,leaves);save('summary.json',summary)
    summary['status']='complete';save('summary.json',summary);print('FINISHED',flush=True)

if __name__=='__main__':main()
