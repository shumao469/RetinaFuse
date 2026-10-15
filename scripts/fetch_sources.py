"""Fetch exact upstream blobs without republishing or relicensing them."""
from pathlib import Path
import json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    subprocess.run(['git','submodule','update','--init','--recursive'],cwd=ROOT,check=True)
    for r in json.loads((ROOT/'provenance/frozen_vendor_sources.json').read_text()):
        raw=subprocess.check_output(['git','-C',str(ROOT/'components'/r['component']),'show',r['commit']+':'+r['upstream_path']])
        assert hashlib.sha256(raw).hexdigest()==r['sha256'],r['upstream_path']
        p=ROOT/r['destination'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    print('PASS: exact pinned upstream model and mask blobs')
if __name__=='__main__':main()
