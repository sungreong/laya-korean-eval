from pathlib import Path
import requests, hashlib, json, time
ROOT=Path(__file__).resolve().parents[1]
rev='1720e3e3357cfe1e281542e223f8273b0890ca34'
dest=ROOT/'evaluation/models/base';dest.mkdir(parents=True,exist_ok=True)
files=['encoder/config.json','rl_agent_config.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json','model.safetensors']
out=[]
expected={r['file']:r for r in json.loads((ROOT/'research/checkpoint-manifest.json').read_text())['files']}
for name in files:
    p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
    if p.is_file() and p.stat().st_size==expected[name]['bytes']:
        with p.open('rb') as existing:
            digest=hashlib.file_digest(existing,'sha256').hexdigest()
        if digest==expected[name]['sha256']:
            out.append(dict(expected[name]))
            print(name,'verified cached file',flush=True)
            continue
    url=f'https://huggingface.co/convaiinnovations/laya-multilingual/resolve/{rev}/{name}'
    with requests.get(url,stream=True,timeout=90) as r:
        r.raise_for_status();h=hashlib.sha256();size=0;t=time.perf_counter()
        partial=p.with_suffix(p.suffix+'.partial')
        with partial.open('wb') as f:
            for chunk in r.iter_content(2**20):f.write(chunk);h.update(chunk);size+=len(chunk)
    if h.hexdigest()!=expected[name]['sha256'] or size!=expected[name]['bytes']:
        raise RuntimeError(f'Checkpoint digest mismatch: {name}; existing file preserved')
    partial.replace(p)
    out.append({'file':name,'sha256':h.hexdigest(),'bytes':size})
    print(name,size,round(time.perf_counter()-t,1),flush=True)
(dest/'download-manifest.json').write_text(json.dumps({'model':'convaiinnovations/laya-multilingual','revision':rev,'files':out},indent=2),encoding='utf-8')
