import hashlib, json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class ProvenanceTests(unittest.TestCase):
    def test_original_execution_hash(self):
        folder=ROOT/'results/pilot-2026-10-05'
        summary=json.loads((folder/'summary.json').read_text())
        digest=hashlib.sha256((folder/'executed-run_evaluation.py').read_bytes()).hexdigest()
        self.assertEqual(digest,summary['script_sha256'])
    def test_prepared_splits_when_available(self):
        data=ROOT/'evaluation/data'
        if not (data/'provenance.json').exists():self.skipTest('Run prepare to validate downloaded data')
        expected=json.loads((ROOT/'results/pilot-2026-10-05/data-provenance.json').read_text())
        groups=[]
        for name,n in [('train',200),('calibration',80),('test',200)]:
            path=data/(name+'.jsonl')
            # Pilot data was prepared on Windows (CRLF), then evaluated in Docker.
            # Normalize line endings to that recorded representation only; content
            # and row order must still match the original byte-level digest.
            recorded_bytes=path.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')
            self.assertEqual(hashlib.sha256(recorded_bytes).hexdigest(),expected['selected_sha256'][path.name])
            rows=[json.loads(line) for line in path.read_text('utf-8').splitlines()]
            self.assertEqual(len(rows),n)
            groups.append({r['state'] for r in rows})
        for a,b in [(0,1),(0,2),(1,2)]:self.assertFalse(groups[a]&groups[b])
