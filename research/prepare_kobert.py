"""Download a fixed KoBERT revision into the ignored evaluation model directory."""
from pathlib import Path
import json
from huggingface_hub import snapshot_download

ROOT=Path(__file__).resolve().parents[1]
REVISION='359874884642d748079d4dd4ff547f2cdbff67d6'
DEST=ROOT/'evaluation/models/kobert-base-v1'

def main():
    snapshot_download(repo_id='skt/kobert-base-v1',revision=REVISION,local_dir=DEST,
                      allow_patterns=['config.json','model.safetensors','spiece.model',
                                      'tokenizer_config.json','special_tokens_map.json'])
    required=['config.json','model.safetensors','spiece.model','tokenizer_config.json','special_tokens_map.json']
    missing=[name for name in required if not (DEST/name).is_file()]
    if missing: raise RuntimeError(f'Missing KoBERT files: {missing}')
    (DEST/'source.json').write_text(json.dumps({'repo':'skt/kobert-base-v1','revision':REVISION,
                                                'files':required},indent=2),encoding='utf-8')
    print('Prepared KoBERT',REVISION,flush=True)

if __name__=='__main__': main()
