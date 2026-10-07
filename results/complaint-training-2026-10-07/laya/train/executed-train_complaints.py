"""Train the LAYA decision head on synthetic 27-way complaint routing data."""
from pathlib import Path
import argparse, gc, hashlib, json, math, os, random, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/laya'))
sys.path.insert(0, str(ROOT / 'research'))
os.environ['USE_TF'] = '0'; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
import laya
from laya.train import TrainConfig, items_from_rows, save_checkpoint, train_model
from metrics import metrics


def load_rows(name):
    path = ROOT / 'datasets/complaints/training' / name
    return [json.loads(x) for x in path.read_text('utf-8').splitlines() if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--out', default='evaluation/complaint-training-e5')
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists(): raise RuntimeError('Choose a fresh output directory')
    out.mkdir(parents=True)
    train, validation = load_rows('train.jsonl'), load_rows('validation.jsonl')
    assert not {r['state'] for r in train} & {r['state'] for r in validation}
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    agent = laya.load(str(ROOT / 'evaluation/models/base'), device='cpu', compile=False)
    agent.temperature = [1., 1., 1.]; agent.temperature_by_options = {}
    items, skipped = items_from_rows(agent.tok, train, 2048, 1536)
    if skipped or len(items) != len(train): raise RuntimeError(f'Training conversion failed: {skipped}')
    summary = {'status': 'running', 'data': 'datasets/complaints/training', 'train_rows': len(train),
               'validation_rows': len(validation), 'epochs_planned': args.epochs, 'selected_epoch': None,
               'settings': {'freeze_encoder': True, 'loss': 'soft-ce', 'head_lr': 1e-4,
                            'micro_batch': 4, 'grad_accum': 4, 'max_len': 2048, 'head_max_len': 1536,
                            'seed': 42, 'temperature_calibration': False, 'device': 'cpu', 'threads': 4},
               'epoch_history': [], 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / 'executed-train_complaints.py').write_bytes(Path(__file__).read_bytes())

    def save(): (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    save()

    def evaluate(epoch):
        agent.model.eval(); rows=[]
        for i, item in enumerate(validation, 1):
            tick=time.perf_counter(); answer=agent.predict(item['state'], item['questions'], max_len=2048, head_max_len=1536)['answers']['route']
            expected=max(item['gold']['route']['probabilities'], key=item['gold']['route']['probabilities'].get)
            rows.append({'id':item['id'],'expected':expected,'predicted':answer['choice'],'probabilities':answer['probabilities'],
                         'confidence':answer['answer_confidence'],'latency_ms':(time.perf_counter()-tick)*1000})
            if i % 54 == 0: print('validation',epoch,i,'/',len(validation),flush=True)
        measured=metrics(rows); measured['nll']=float(np.mean([-math.log(max(r['probabilities'][r['expected']],1e-9)) for r in rows]))
        (out/f'validation_epoch_{epoch}_predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        return measured

    best=(-1.,-float('inf')); best_state=None; start=time.perf_counter()
    def after_epoch(epoch, loss):
        nonlocal best,best_state
        rng_state=torch.random.get_rng_state(); measured=evaluate(epoch+1); key=(measured['accuracy'],-measured['nll'])
        if key>best:
            best=key; summary['selected_epoch']=epoch+1
            best_state={n:t.detach().cpu().clone() for n,t in agent.model.state_dict().items() if not n.startswith('encoder.')}
            torch.save(best_state,out/'best-head-fp32.pt')
        summary['epoch_history'].append({'epoch':epoch+1,'train_loss':loss,'validation':measured}); save()
        print('epoch',epoch+1,'validation accuracy',measured['accuracy'],'nll',measured['nll'],flush=True)
        agent.model.train(); agent.model.encoder.eval(); torch.random.set_rng_state(rng_state)

    cfg=TrainConfig(epochs=args.epochs,micro_batch=4,grad_accum=4,head_lr=1e-4,loss='soft-ce',
                    shuffle_options=('choice',),freeze_encoder=True,seed=42,amp=False,gradient_checkpointing=False,
                    log_every=50,max_len=2048,head_max_len=1536)
    losses=train_model(agent.model,agent.tok,items,cfg,torch.device('cpu'),2048,1536,on_epoch_end=after_epoch)
    summary['training_and_validation_seconds']=time.perf_counter()-start; summary['epoch_losses']=losses
    agent.model.load_state_dict(best_state,strict=False); agent.model.eval()
    checkpoint=out/'selected-model'; config=dict(agent.cfg,temperature=[1.,1.,1.],fine_tuned=True,max_len=2048,head_max_len=1536)
    config.pop('temperature_by_options',None); save_checkpoint(agent.model,agent.tok,config,str(checkpoint))
    del agent,best_state; gc.collect()
    # Verify the actual artifact can be loaded before marking the run complete.
    loaded=laya.load(str(checkpoint),device='cpu',compile=False); del loaded; gc.collect()
    summary['selected_checkpoint']=str(checkpoint.relative_to(ROOT)); summary['export_dtype']='float16'
    summary['status']='complete'; summary['total_seconds']=time.perf_counter()-start; save()
    print('FINISHED selected epoch',summary['selected_epoch'],flush=True)


if __name__=='__main__': main()
