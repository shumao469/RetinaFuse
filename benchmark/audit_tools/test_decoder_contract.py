"""CPU-light synthetic contract tests of the frozen decoder, not study results."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def functions(path, names, env):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    vendor = ROOT / 'vendor/retinafuse_core.py'
    decoder = ROOT / 'code/regve.py'
    env = dict(np=np, ORIG_H=1216, ORIG_W=1936, IMG_H=64, IMG_W=64,
               TIME_WINDOW=.02, SEQ_LEN=60)
    functions(vendor, {'build_event_frames', 'extract_stat_features'}, env)
    company = SimpleNamespace(build_event_frames=env['build_event_frames'],
                              extract_stat_features=env['extract_stat_features'])
    env = dict(np=np, company=company)
    functions(decoder, {'convert'}, env)
    convert = env['convert']
    # Only one pixel varies, and exported duplicates retain their physical timestamps.
    gray = np.full((5, 8, 10), 30, np.uint8)
    gray[2:4, 3, 4] = 90
    times = np.arange(5) / 4.
    mask = np.ones((8, 10), bool)
    frames4, stats4, qc4 = convert(gray, times, mask, 4., .02)
    frames2, stats2, qc2 = convert(gray, times, mask, 2., .02)
    framesd, statsd, qcd = convert(gray, times, mask, 4., .02, dedup=True)
    assert qc4['selected_frames'] == 5 and qc2['selected_frames'] == 3
    assert qcd['selected_frames'] == 3
    assert qc4['events'] == qc2['events'] == qcd['events'] == 2
    np.testing.assert_array_equal(frames4, frames2)
    np.testing.assert_array_equal(frames4, framesd)
    np.testing.assert_array_equal(stats4, stats2)
    np.testing.assert_array_equal(stats4, statsd)
    np.testing.assert_allclose(stats4[:7], [2, 1, 1, .5, .5, 0, .5])
    assert frames4[:, 0].sum() == frames4[:, 1].sum() == 1
    assert frames4.sum() == stats4[0]
    masked = mask.copy(); masked[3, 4] = False
    f, s, q = convert(gray, times, masked, 4., .02)
    assert q['empty'] and not f.any() and not s.any()
    f, s, q = convert(gray, times, mask, 4., 2.)
    assert q['empty'] and not f.any() and not s.any()
    # Two simultaneous changes yield zero inter-event intervals, not sub-frame measurements.
    gray[2:4, 2, 2] = 90
    f, s, q = convert(gray, times, mask, 4., .02)
    assert q['events'] == 4 and f.sum() == 4
    np.testing.assert_allclose(s[4], .5 / 3, rtol=1e-6)
    result = dict(status='PASS', fixture='synthetic unit tests; not animal data or efficacy results',
                  checks=['sampling indices', 'physical time retained after duplicate removal',
                          'positive/negative grayscale-change counts', 'event-count conservation',
                          'mask support', 'threshold dependence', 'empty-input behavior',
                          'simultaneous events and zero intervals'],
                  decoder_sha256=digest(decoder), vendor_sha256=digest(vendor),
                  note='The decoder source_fps field denotes requested conversion sampling, not AVI container fps.')
    (ROOT / 'audit/decoder_contract_tests.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='PASS', checks=len(result['checks']))))

if __name__ == '__main__':
    main()
