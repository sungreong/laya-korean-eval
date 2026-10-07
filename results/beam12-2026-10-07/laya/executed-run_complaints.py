"""Evaluate flat, hard-cascade, and top-k beam complaint routing; synthetic cases only."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research/laya'))
os.environ['USE_TF'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import torch
import laya
from sklearn.metrics import f1_score
from metrics import wilson
from complaint_schema import build_questions, question


def summarize(rows, leaves):
    strict = [r for r in rows if not r['needs_clarification']]
    ambiguous = [r for r in rows if r['needs_clarification']]

    def score(items):
        correct = sum(r['predicted_leaf'] == r['expected_leaf'] for r in items)
        prefix = [sum(r['predicted_path'][:d] == r['expected_path'][:d] for r in items) / len(items) for d in [1, 2, 3]]
        errors = {'major': 0, 'middle': 0, 'leaf': 0}
        for r in items:
            for depth, name in enumerate(errors):
                if r['predicted_path'][depth] != r['expected_path'][depth]:
                    errors[name] += 1
                    break
        return {'n': len(items), 'correct_paths': correct, 'major_accuracy': prefix[0],
                'middle_path_accuracy': prefix[1], 'exact_path_accuracy': prefix[2],
                'wilson95_exact': wilson(correct, len(items)), 'first_error_level': errors,
                'leaf_macro_f1': float(f1_score([r['expected_leaf'] for r in items], [r['predicted_leaf'] for r in items],
                                              labels=list(leaves), average='macro', zero_division=0))}

    result = {'strict': score(strict),
              'by_slice': {s: score([r for r in strict if r['slice'] == s]) for s in sorted({r['slice'] for r in strict})},
              'under_specified': {'n': len(ambiguous), 'acceptable_path_hits': sum(r['predicted_leaf'] in r['acceptable_leaves'] for r in ambiguous),
                                  'note': 'Not single-answer accuracy. All these cases require clarification; the classifier was forced to choose.'},
              'latency_ms_p50': float(np.percentile([r['latency_ms'] for r in rows], 50)),
              'latency_ms_p95': float(np.percentile([r['latency_ms'] for r in rows], 95)),
              'cases_with_input_or_option_truncation': sum(any(t['usage'].get('truncated', False) or t['usage'].get('truncated_questions') for t in r['trace']) for r in rows)}
    if strict and all('novelty' in row for row in strict):
        result['by_novelty'] = {kind: score([row for row in strict if row['novelty'] == kind])
                                for kind in sorted({row['novelty'] for row in strict})}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='evaluation/models/base')
    parser.add_argument('--out', default='evaluation/complaints-base')
    parser.add_argument('--fixture', default='datasets/complaints')
    parser.add_argument('--methods', nargs='+', choices=['flat27','cascade','cascade_prefix','beam12'], default=['flat27','cascade','cascade_prefix'])
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError('Choose a fresh output directory; existing results are preserved')
    out.mkdir(parents=True)
    fixture = ROOT / args.fixture
    taxonomy = json.loads((fixture / 'taxonomy.json').read_text('utf-8'))
    cases = json.loads((fixture / 'cases.json').read_text('utf-8'))
    leaves, majors, q = build_questions(taxonomy)
    torch.set_num_threads(4); torch.set_num_interop_threads(1); torch.manual_seed(42)
    agent = laya.load(str(ROOT / args.model), device='cpu', compile=False)
    agent.temperature = [1., 1., 1.]; agent.temperature_by_options = {}
    beam_descriptions = {
        leaf_id: ' > '.join(leaf['path']) + ': ' + leaf['description']
        for leaf_id, leaf in leaves.items()
    }
    descriptions = list(q['flat'].values()) + list(q['top'].values()) + list(beam_descriptions.values())
    descriptions += [v for x in q['middle'].values() for v in x.values()]
    lengths = [len(agent.tok.encode(s, add_special_tokens=False)) for s in descriptions]
    if max(lengths) > 48:
        raise RuntimeError(f'Option description exceeds upstream per-option 48-token cap: {max(lengths)}')

    summary = {'status': 'running', 'model': args.model, 'fixture': args.fixture,
               'fixture_sha256': {n: hashlib.sha256((fixture / n).read_bytes()).hexdigest() for n in ['taxonomy.json','cases.json']},
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'counts': {'major': len(majors), 'middle': sum(len(x['children']) for x in majors.values()),
                          'leaf': len(leaves), 'strict_cases': sum(not x['needs_clarification'] for x in cases),
                          'under_specified': sum(x['needs_clarification'] for x in cases)},
               'settings': {'max_len': 2048, 'head_max_len': 1536, 'max_option_tokens': max(lengths),
                            'device': 'cpu', 'threads': 4, 'temperature': 1., 'option_order': 'fixed taxonomy order'},
               'scope': 'Synthetic author-defined diagnostic; no independent human review; not a production benchmark.',
               'methods': {}}

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    (out / 'executed-run_complaints.py').write_bytes(Path(__file__).read_bytes())
    save('questions.json', q)
    save('summary.json', summary)

    def ask(state, criteria, stage):
        response = agent.predict(state, question(criteria), max_len=2048, head_max_len=1536)
        answer = response['answers']['route']
        if len(answer['probabilities']) != len(criteria):
            raise RuntimeError('Candidate count changed during encoding')
        return answer['choice'], {'stage': stage, 'candidate_count': len(criteria), 'answer': answer, 'usage': response.get('usage', {})}

    def top_keys(trace, k):
        probabilities = trace['answer']['probabilities']
        return sorted(probabilities, key=probabilities.get, reverse=True)[:k]

    def predict(case, method):
        state = f"상담 경로: {case['channel']}\n민원 요약: {case['summary']}"
        start = time.perf_counter()
        if method == 'flat27':
            leaf, trace = ask(state, q['flat'], 'leaf27')
            traces = [trace]
        elif method == 'cascade':
            major, t1 = ask(state, q['top'], 'major3')
            middle, t2 = ask(state, q['middle'][major], 'middle3')
            leaf, t3 = ask(state, q['bottom'][middle], 'leaf3')
            traces = [t1, t2, t3]
        elif method == 'cascade_prefix':
            major, t1 = ask(state, q['top'], 'major3')
            major_name = majors[major]['name']
            middle_state = state + f"\n이전 단계 선택 대분류: {major_name}"
            middle, t2 = ask(middle_state, q['middle'][major], 'middle3')
            middle_name = next(node['name'] for node in majors[major]['children'] if node['id'] == middle)
            leaf_state = middle_state + f"\n이전 단계 선택 중분류: {middle_name}"
            leaf, t3 = ask(leaf_state, q['bottom'][middle], 'leaf3')
            traces = [t1, t2, t3]
        else:
            # Keep two major branches, two middle branches under each parent,
            # then rerank all three leaves from all four paths in one final call.
            _, t1 = ask(state, q['top'], 'major3_beam_top2')
            major_ids = top_keys(t1, 2)
            middle_ids, middle_traces = [], []
            for major_id in major_ids:
                major_state = state + f"\n검토 중인 대분류 후보: {majors[major_id]['name']}"
                _, middle_trace = ask(major_state, q['middle'][major_id], f'middle3_under_{major_id}_top2')
                middle_ids.extend(top_keys(middle_trace, 2))
                middle_traces.append(middle_trace)
            final_ids = [leaf_id for middle_id in middle_ids for leaf_id in q['bottom'][middle_id]]
            if len(final_ids) != 12 or len(set(final_ids)) != 12:
                raise RuntimeError(f'beam12 expected 12 unique leaf candidates, got {len(set(final_ids))}')
            final_criteria = {leaf_id: beam_descriptions[leaf_id] for leaf_id in final_ids}
            leaf, final_trace = ask(state, final_criteria, 'leaf12_full_path_rerank')
            traces = [t1] + middle_traces + [final_trace]
        return dict(case, predicted_leaf=leaf, predicted_path=leaves[leaf]['path'],
                    expected_path=leaves[case['expected_leaf']]['path'] if case['expected_leaf'] else None,
                    acceptable_paths=[leaves[k]['path'] for k in case['acceptable_leaves']],
                    latency_ms=(time.perf_counter()-start)*1000, trace=traces)

    for method in args.methods:
        predict(cases[0], method)  # Per-method warm-up excluded from latency measurements.
        rows = []
        for i, case in enumerate(cases, 1):
            rows.append(predict(case, method))
            if i % 10 == 0:
                save(method + '_predictions.json', rows)
                print(method, i, '/', len(cases), flush=True)
        save(method + '_predictions.json', rows)
        summary['methods'][method] = summarize(rows, leaves)
        save('summary.json', summary)
        print(method, json.dumps(summary['methods'][method], ensure_ascii=False), flush=True)
    summary['status'] = 'complete'
    save('summary.json', summary)
    print('FINISHED', flush=True)


if __name__ == '__main__':
    main()
