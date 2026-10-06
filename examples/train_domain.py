"""Fine-tune local Laya with externally separated train/calibration/test JSONL.

Pinned source: Laya 8a6e1328. This CLI trains and calibrates; use the evaluation
harness to score test data. No Hub uploads and no mutation of the base checkpoint.
"""
from pathlib import Path
import argparse, json, shutil, tempfile
import torch
from laya.train import TrainConfig,load_checkpoint,read_jsonl,items_from_rows,train_model,calibration_records,save_checkpoint
from laya.calibrate import fit_temperature_map

def main():
    ap=argparse.ArgumentParser()
    for k in ['base','train','calibration','test','out']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--epochs',type=int,default=2)
    ap.add_argument('--freeze-encoder',action='store_true')
    args=ap.parse_args()
    base,out=Path(args.base).resolve(),Path(args.out).resolve()
    if out==base or base in out.parents or out in base.parents:raise ValueError('Use a separate output directory outside the base checkpoint.')
    if out.exists():raise ValueError('Choose a new output directory; existing checkpoints will not be overwritten.')
    groups=[read_jsonl(p) for p in [args.train,args.calibration,args.test]]
    keys=[{json.dumps(r['state'],ensure_ascii=False,sort_keys=True) for r in rows} for rows in groups]
    if any(keys[a]&keys[b] for a,b in [(0,1),(0,2),(1,2)]):raise ValueError('Identical states cross split boundaries.')
    # Upstream load_checkpoint may normalize tokenizer config. Work on a temporary copy.
    with tempfile.TemporaryDirectory(prefix='laya-load-') as tmp:
        copied=Path(tmp)/'base';shutil.copytree(base,copied)
        model,tok,cfg=load_checkpoint(str(copied))
    max_len,head_max_len=cfg.get('max_len',1024),cfg.get('head_max_len',256)
    train,skips=items_from_rows(tok,groups[0],max_len,head_max_len)
    calib,cskips=items_from_rows(tok,groups[1],max_len,head_max_len)
    if skips or cskips:raise ValueError(f'Invalid or truncated options: {skips}, {cskips}')
    if len(calib)<10:raise ValueError('At least 10 calibration questions are needed; use more for meaningful evaluation.')
    config=TrainConfig(epochs=args.epochs,micro_batch=1,grad_accum=16,head_lr=5e-5,encoder_lr=1e-5,loss='soft-ce',shuffle_options=('choice',),freeze_encoder=args.freeze_encoder,seed=42)
    device=torch.device(args.device)
    losses=train_model(model,tok,train,config,device,max_len,head_max_len)
    fitted=fit_temperature_map(calibration_records(model,tok,calib,device,max_len,head_max_len,batch_size=1))
    cfg.update(fine_tuned=True,temperature=fitted['temperature'])
    cfg.pop('temperature_by_options',None)
    if fitted['temperature_by_options']:cfg['temperature_by_options']=fitted['temperature_by_options']
    save_checkpoint(model,tok,cfg,str(out))
    (out/'training-report.json').write_text(json.dumps({'losses':losses,'calibration':fitted,'test_rows_not_used_for_training':len(groups[2]),'note':'Run held-out evaluation separately. Exported weights are fp16.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved:',out)

if __name__=='__main__':main()
