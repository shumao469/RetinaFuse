"""Prepare a fresh run from an existing cache without changing the original experiment."""
from pathlib import Path
import argparse,json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir',required=True);ap.add_argument('--cached-data',required=True);args=ap.parse_args()
    dest=Path(args.workdir).resolve();cache=Path(args.cached_data).resolve()
    if dest.exists() and any(dest.iterdir()):raise RuntimeError('Choose a fresh empty working directory')
    if dest==cache or dest in cache.parents:raise RuntimeError('Work directory must not contain the source cache')
    dest.mkdir(parents=True,exist_ok=True);shutil.copytree(cache,dest/'data')
    for d in ['code','audit_tools','followup']:shutil.copytree(ROOT/'benchmark'/d,dest/d)
    shutil.copytree(ROOT/'benchmark/vendor',dest/'vendor')
    (dest/'protocol.json').write_text((ROOT/'benchmark/protocol.json').read_text(),encoding='utf-8')
    specpath=dest/'followup/gate_protocol.json'
    spec=json.loads(specpath.read_text(encoding='utf-8'))
    spec['original_training_protocol_sha256']=spec['training_protocol_sha256']
    spec['training_protocol_sha256']=hashlib.sha256((dest/'protocol.json').read_bytes()).hexdigest()
    specpath.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    print('Prepared new run:',dest)
if __name__=='__main__':main()
