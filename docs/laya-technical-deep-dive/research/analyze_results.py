from pathlib import Path
import json,numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'evaluation/results'
def read(name):return json.loads((R/name).read_text('utf-8'))
pairs=[]
for a,b in [('base_ko','base_en'),('base_ko','finetuned_raw'),('tfidf_logreg','base_ko')]:
    aa=read(('baseline' if a=='tfidf_logreg' else a)+'_predictions.json');bb=read(b+'_predictions.json')
    assert [r['id'] for r in aa]==[r['id'] for r in bb]
    ga=np.array([r['predicted']==r['expected'] for r in aa],dtype=int);gb=np.array([r['predicted']==r['expected'] for r in bb],dtype=int)
    onlya=int(((ga==1)&(gb==0)).sum());onlyb=int(((ga==0)&(gb==1)).sum())
    rng=np.random.default_rng(42);d=gb-ga;boot=d[rng.integers(0,len(d),size=(20000,len(d)))].mean(axis=1)
    pairs.append({'a':a,'b':b,'delta_b_minus_a':float(d.mean()),'paired_bootstrap95':list(map(float,np.percentile(boot,[2.5,97.5]))),'a_only_correct':onlya,'b_only_correct':onlyb,'mcnemar_exact_p_unadjusted':float(binomtest(onlyb,onlya+onlyb,.5).pvalue) if onlya+onlyb else 1.})
out={'note':'Exploratory paired analyses, unadjusted p values; not confirmatory multi-hypothesis inference. Bootstrap resamples rows, 20000 replicates, seed42.','pairs':pairs}
(R/'paired-analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
