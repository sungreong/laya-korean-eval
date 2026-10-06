"""Larger real-data training, validation-only epoch selection, exported-model test."""
from pathlib import Path
import argparse, csv, gc, hashlib, json, math, os, platform, random, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/laya'))
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import psutil
import torch
import laya
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from laya.train import TrainConfig, items_from_rows, save_checkpoint, train_model
from metrics import metrics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]


def gold(row):
    return max(row['gold']['sentiment']['probabilities'], key=row['gold']['sentiment']['probabilities'].get)


def prepare(data, out, size, seed):
    old_train = read_jsonl(data / 'train.jsonl')
    calibration = read_jsonl(data / 'calibration.jsonl')
    old_test = read_jsonl(data / 'test.jsonl')
    questions = json.loads((data / 'questions.json').read_text('utf-8'))
    raw = {}
    for split in ['train', 'test']:
        with (data / f'ratings_{split}.txt').open(encoding='utf-8', newline='') as f:
            raw[split] = list(csv.DictReader(f, delimiter='\t'))
    forbidden = {r['state'] for r in old_train + calibration + old_test}
    test_texts = {r['document'].strip() for r in raw['test']}
    rng = random.Random(seed)

    def sample(pool, n, blocked):
        rows, seen = [], set(blocked)
        for row in pool:
            state = row['document'].strip()
            if not state or state in seen:
                continue
            seen.add(state)
            label = row['label'] == '1'
            rows.append({'id': row['id'], 'state': state, 'questions': questions['ko'],
                         'gold': {'sentiment': {'probabilities': {'A': float(label), 'B': float(not label)}}}})
        selected = []
        for label in ['A', 'B']:
            selected.extend(rng.sample([r for r in rows if gold(r) == label], n // 2))
        rng.shuffle(selected)
        return selected

    extra = sample(raw['train'], size - len(old_train), forbidden | test_texts)
    train = old_train + extra
    rng.shuffle(train)
    validation = sample(raw['train'], 200, forbidden | test_texts | {r['state'] for r in train})
    new_test = sample(raw['test'], 500, forbidden | {r['state'] for r in train + validation})
    splits = {'train': train, 'validation': validation, 'calibration_unused': calibration,
              'original_test': old_test, 'new_test': new_test}
    groups = [set(r['state'] for r in rows) for rows in splits.values()]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not a & b, 'Split leakage'
    assert len(train) == size and len({r['state'] for r in train}) == size
    provenance = {'seed': seed, 'splits': {}, 'raw_sha256': {s: hashlib.sha256((data / f'ratings_{s}.txt').read_bytes()).hexdigest() for s in raw}}
    for name, rows in splits.items():
        payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (out / f'{name}.jsonl').write_text(payload, encoding='utf-8', newline='\n')
        provenance['splits'][name] = {'n': len(rows), 'ids': [r['id'] for r in rows],
                                     'positive': sum(gold(r) == 'A' for r in rows),
                                     'sha256_utf8_lf': hashlib.sha256(payload.encode()).hexdigest()}
    return splits, questions, provenance


def paired(a, b):
    assert [r['id'] for r in a] == [r['id'] for r in b]
    aa = np.array([r['predicted'] == r['expected'] for r in a], dtype=int)
    bb = np.array([r['predicted'] == r['expected'] for r in b], dtype=int)
    delta = bb - aa
    rng = np.random.default_rng(42)
    boot = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(axis=1)
    only_a, only_b = int(((aa == 1) & (bb == 0)).sum()), int(((aa == 0) & (bb == 1)).sum())
    return {'delta': float(delta.mean()), 'bootstrap95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'base_only_correct': only_a, 'trained_only_correct': only_b,
            'mcnemar_p_unadjusted': float(binomtest(only_b, only_a + only_b).pvalue) if only_a + only_b else 1.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--out', default='evaluation/scale-1000-e5')
    args = parser.parse_args()
    if args.train_size < 200 or args.train_size % 2 or args.epochs < 1:
        parser.error('train-size must be even and >=200; epochs must be positive')
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError('Choose a new output directory; existing runs are preserved')
    out.mkdir(parents=True)
    data = ROOT / 'evaluation/data'
    splits, questions, provenance = prepare(data, out, args.train_size, 42)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    start = time.perf_counter()
    summary = {'status': 'running', 'protocol': 'research/scaling-protocol.md',
               'train_size': args.train_size, 'epochs_planned': args.epochs, 'seed': 42,
               'freeze_encoder': True, 'head_lr': 1e-4, 'temperature_calibration': False,
               'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform(),
                               'threads': 4, 'device': 'cpu', 'training_dtype': 'float32', 'export_dtype': 'float16'},
               'epoch_history': [], 'stages': {}}

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    save('provenance.json', provenance)
    (out / 'executed-run_scaling.py').write_bytes(Path(__file__).read_bytes())
    summary['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save('summary.json', summary)
    agent = laya.load(str(ROOT / 'evaluation/models/base'), device='cpu', compile=False)
    summary['load_seconds'] = time.perf_counter() - start
    agent.temperature = [1., 1., 1.]
    agent.temperature_by_options = {}

    def evaluate(name, rows, q, diagnostic=False):
        agent.model.eval()
        result = []
        for i, row in enumerate(rows):
            tick = time.perf_counter()
            answer = agent.predict(row['state'], q)['answers']['department' if diagnostic else 'sentiment']
            result.append({'id': row['id'], 'expected': row['expected'] if diagnostic else gold(row),
                           'predicted': answer['choice'], 'probabilities': answer['probabilities'],
                           'confidence': answer['answer_confidence'], 'latency_ms': (time.perf_counter() - tick) * 1000})
            if (i + 1) % 100 == 0:
                print(name, i + 1, flush=True)
        measured = metrics(result)
        measured['nll'] = float(np.mean([-math.log(max(r['probabilities'][r['expected']], 1e-9)) for r in result]))
        summary['stages'][name] = measured
        save(name + '_predictions.json', result)
        save('summary.json', summary)
        print(name, 'accuracy', measured['accuracy'], 'nll', measured['nll'], flush=True)
        return result

    # Test measurements are recorded, but never used by training or epoch selection.
    baseline = {name: evaluate('base_' + name, splits[name], questions['ko']) for name in ['original_test', 'new_test']}
    diagnostics = read_jsonl(data / 'diagnostics.jsonl')
    evaluate('base_diagnostic', diagnostics, questions['department'], diagnostic=True)
    evaluate('base_validation', splits['validation'], questions['ko'])

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=1)
    classifier = LogisticRegression(C=1, max_iter=1000, random_state=42)
    classifier.fit(vectorizer.fit_transform([r['state'] for r in splits['train']]), [gold(r) for r in splits['train']])
    for name in ['original_test', 'new_test']:
        result = []
        for row in splits[name]:
            tick = time.perf_counter()
            p = classifier.predict_proba(vectorizer.transform([row['state']]))[0]
            probabilities = dict(zip(classifier.classes_, map(float, p)))
            prediction = max(probabilities, key=probabilities.get)
            result.append({'id': row['id'], 'expected': gold(row), 'predicted': prediction, 'probabilities': probabilities,
                           'confidence': probabilities[prediction], 'latency_ms': (time.perf_counter() - tick) * 1000})
        summary['stages']['tfidf_' + name] = metrics(result)
        save('tfidf_' + name + '_predictions.json', result)
    save('summary.json', summary)

    items, skipped = items_from_rows(agent.tok, splits['train'], 1024, 256)
    assert not skipped
    best_key = (-1., -float('inf'))
    best_state = None
    training_start = time.perf_counter()

    def after_epoch(epoch, loss):
        nonlocal best_key, best_state
        elapsed_before_validation = time.perf_counter() - training_start
        rng_state = torch.random.get_rng_state()
        evaluate(f'validation_epoch_{epoch + 1}', splits['validation'], questions['ko'])
        measured = summary['stages'][f'validation_epoch_{epoch + 1}']
        key = (measured['accuracy'], -measured['nll'])
        if key > best_key:
            best_key = key
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in agent.model.state_dict().items()
                          if not name.startswith('encoder.')}
            summary['selected_epoch'] = epoch + 1
            torch.save(best_state, out / 'best-head-fp32.pt')
        summary['epoch_history'].append({'epoch': epoch + 1, 'train_loss': loss,
                                         'validation_accuracy': measured['accuracy'], 'validation_nll': measured['nll'],
                                         'elapsed_training_and_prior_validation_seconds': elapsed_before_validation})
        save('summary.json', summary)
        # predict() switches to eval; restore the training state without resetting optimizer/scheduler.
        agent.model.train()
        agent.model.encoder.eval()
        torch.random.set_rng_state(rng_state)

    config = TrainConfig(epochs=args.epochs, micro_batch=4, grad_accum=4, head_lr=1e-4,
                         loss='soft-ce', shuffle_options=('choice',), freeze_encoder=True, seed=42,
                         amp=False, gradient_checkpointing=False, log_every=50, max_len=1024, head_max_len=256)
    losses = train_model(agent.model, agent.tok, items, config, torch.device('cpu'), 1024, 256, on_epoch_end=after_epoch)
    summary['training_and_validation_seconds'] = time.perf_counter() - training_start
    summary['epoch_losses'] = losses
    assert best_state is not None
    agent.model.load_state_dict(best_state, strict=False)
    agent.model.eval()
    checkpoint = out / 'selected-model'
    cfg = dict(agent.cfg, temperature=[1., 1., 1.], fine_tuned=True)
    cfg.pop('temperature_by_options', None)
    save_checkpoint(agent.model, agent.tok, cfg, str(checkpoint))
    del best_state, agent
    gc.collect()
    agent = laya.load(str(checkpoint), device='cpu', compile=False)
    summary['test_model'] = 'selected-model exported FP16 and reloaded'
    summary['selected_checkpoint'] = str(checkpoint.relative_to(ROOT))
    for name in ['original_test', 'new_test']:
        trained = evaluate('trained_' + name, splits[name], questions['ko'])
        summary.setdefault('paired', {})[name] = paired(baseline[name], trained)
    evaluate('trained_diagnostic', diagnostics, questions['department'], diagnostic=True)
    summary['status'] = 'complete'
    summary['total_seconds'] = time.perf_counter() - start
    summary['rss_mib'] = psutil.Process().memory_info().rss / 2**20
    save('summary.json', summary)
    print('FINISHED', json.dumps({'selected_epoch': summary['selected_epoch'], 'paired': summary['paired']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
