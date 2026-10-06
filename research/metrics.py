"""Classification metrics shared by the evaluation runner and unit tests."""
import math
import numpy as np
from sklearn.metrics import f1_score

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
