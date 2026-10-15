from pathlib import Path
import hashlib,json,os,tempfile
ROOT=Path(os.environ.get('RETINAFUSE_WORKDIR',str(Path(__file__).resolve().parent.parent))).resolve()
CFG=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,p)
def ensure():
    for d in ['data','audit','runs','results','logs','vendor']:(ROOT/d).mkdir(exist_ok=True)
