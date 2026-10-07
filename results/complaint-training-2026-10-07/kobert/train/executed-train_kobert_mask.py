"""Fine-tune a LAYA-like KoBERT mask-marker decision head on complaint routing."""
from pathlib import Path
import argparse, hashlib, json, math, os, random, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research'))
import numpy as np
import torch
from sklearn.metrics import f1_score
from kobert_mask_model import KoBertMaskDecision,collate,pack_one,save_head,load_trained

def rows(name): return [json.loads(x) for x in (ROOT/'datasets/complaints/training'/name).read_text('utf-8').splitlines()]
def target(row): return max(row['gold']['route']['probabilities'],key=row['gold']['route']['probabilities'].get)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--head-lr',type=float,default=1e-3)
    ap.add_argument('--head-layers',type=int,default=0)
    ap.add_argument('--out',default='evaluation/kobert-mask-complaints-e5');args=ap.parse_args()
    out=ROOT/args.out
    if out.exists(): raise RuntimeError('Choose a fresh output directory')
    out.mkdir(parents=True);train,valid=rows('train.jsonl'),rows('validation.jsonl');model_dir=ROOT/'evaluation/models/kobert-base-v1'
    torch.set_num_threads(4);torch.set_num_interop_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    model=KoBertMaskDecision(model_dir,head_layers=args.head_layers);model.encoder.requires_grad_(False);model.encoder.eval()
    tokenizer=model.tokenizer;device=torch.device('cpu');model.to(device)
    summary={'status':'running','model':'skt/kobert-base-v1','revision':'359874884642d748079d4dd4ff547f2cdbff67d6',
             'architecture':f'KoBERT + {args.head_layers} Transformer head layers + shared scorer over real [MASK] positions',
             'train_rows':len(train),'validation_rows':len(valid),'epochs_planned':args.epochs,'selected_epoch':None,
             'settings':{'freeze_encoder':True,'head_layers':args.head_layers,'batch_size':8,'grad_accum':2,'head_lr':args.head_lr,'max_length':512,'seed':42},
             'epoch_history':[],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'executed-train_kobert_mask.py').write_bytes(Path(__file__).read_bytes())
    def save(): (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    save();parameters=list(model.head.parameters())+list(model.scorer.parameters());optimizer=torch.optim.AdamW(parameters,lr=args.head_lr,weight_decay=.01)
    updates=math.ceil(math.ceil(len(train)/8)/2)*args.epochs;scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=updates,eta_min=args.head_lr/100)
    best=(-1.,-float('inf'));best_state=None;start=time.perf_counter();rng=random.Random(42)

    def evaluate(epoch):
        model.eval();predictions=[];losses=[];lengths=[];dropped=0;option_dropped=0
        with torch.no_grad():
            for start_i in range(0,len(valid),8):
                chunk=valid[start_i:start_i+8];packed=[];targets=[]
                for row in chunk:
                    q=row['questions']['route'];p=pack_one(tokenizer,row['state'],q['instructions'],q['criteria']);packed.append(p)
                    targets.append(p['keys'].index(target(row)));lengths.append(len(p['ids']));dropped+=p['state_tokens_dropped']>0
                    option_dropped+=p['option_tokens_dropped']
                batch=collate(packed,tokenizer.pad_token_id,targets);logits=model(**{k:v for k,v in batch.items() if k!='targets'})
                loss=torch.nn.functional.cross_entropy(logits,batch['targets']);losses.append(float(loss))
                probs=torch.softmax(logits,-1).cpu().numpy()
                for row,p,values,t in zip(chunk,packed,probs,targets):
                    pred=int(values.argmax());predictions.append({'id':row['id'],'expected':p['keys'][t],'predicted':p['keys'][pred],
                        'probabilities':dict(zip(p['keys'],map(float,values))),'confidence':float(values[pred]),'latency_ms':0.})
        acc=sum(r['expected']==r['predicted'] for r in predictions)/len(predictions)
        result={'n':len(predictions),'accuracy':acc,'macro_f1':float(f1_score([r['expected'] for r in predictions],[r['predicted'] for r in predictions],average='macro',zero_division=0)),
                'nll':float(np.mean(losses)),'max_tokens':max(lengths),'mean_tokens':float(np.mean(lengths)),
                'truncated_states':dropped,'option_tokens_dropped':option_dropped}
        (out/f'validation_epoch_{epoch}_predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8');return result

    summary['validation_before_training']=evaluate(0);save()
    print('validation before training',summary['validation_before_training'],flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();model.encoder.eval();order=list(range(len(train)));random.Random(42+epoch).shuffle(order);optimizer.zero_grad(set_to_none=True);loss_sum=steps=0
        for batch_n,start_i in enumerate(range(0,len(order),8),1):
            packed=[];targets=[]
            for idx in order[start_i:start_i+8]:
                row=train[idx];q=row['questions']['route'];perm=list(range(len(q['criteria'])));rng.shuffle(perm)
                p=pack_one(tokenizer,row['state'],q['instructions'],q['criteria'],order=perm);packed.append(p);targets.append(p['keys'].index(target(row)))
            batch=collate(packed,tokenizer.pad_token_id,targets);logits=model(**{k:v for k,v in batch.items() if k!='targets'})
            loss=torch.nn.functional.cross_entropy(logits,batch['targets']);(loss/2).backward();loss_sum+=float(loss.detach());steps+=1
            if batch_n%2==0 or start_i+8>=len(order):
                torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            if batch_n%25==0: print('epoch',epoch,'batch',batch_n,'loss',round(float(loss.detach()),4),flush=True)
        measured=evaluate(epoch);mean_loss=loss_sum/steps;key=(measured['accuracy'],-measured['nll'])
        if key>best:
            best=key;summary['selected_epoch']=epoch;best_state=model.trainable_state();torch.save(best_state,out/'best-head-fp32.pt')
        summary['epoch_history'].append({'epoch':epoch,'train_loss':mean_loss,'validation':measured});save()
        print('epoch',epoch,'mean',mean_loss,'validation',measured,flush=True)
    model.load_state_dict(best_state,strict=False);checkpoint=out/'selected-model';save_head(model,checkpoint,{'base_revision':summary['revision'],'max_length':512})
    del model;reloaded,_=load_trained(model_dir,checkpoint);reloaded.eval()
    summary['status']='complete';summary['selected_checkpoint']=str(checkpoint.relative_to(ROOT));summary['total_seconds']=time.perf_counter()-start;save()
    print('FINISHED selected epoch',summary['selected_epoch'],flush=True)

if __name__=='__main__': main()
