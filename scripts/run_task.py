"""Run one prespecified CUDA outer task; no test-driven model search."""
from pathlib import Path
import argparse,json,os,subprocess,sys
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir',required=True);ap.add_argument('--model',default='RetinaFuse');ap.add_argument('--condition',default='graychange_4fps_t002');ap.add_argument('--fold',type=int,choices=range(1,6),required=True);ap.add_argument('--seed',type=int,choices=[42,1042,2042],required=True);a=ap.parse_args()
    work=Path(a.workdir).resolve();assert (work/'data/animal_manifest.csv').exists()
    spec=dict(id=f'fivefold_{a.fold}_{a.model}_{a.condition}_seed{a.seed}',model=a.model,design='fivefold',held_out=a.fold,partition_index=a.fold,seed=a.seed,condition=a.condition)
    env=os.environ.copy();env['RETINAFUSE_WORKDIR']=str(work)
    subprocess.run([sys.executable,str(work/'code/train.py'),'--task',json.dumps(spec)],cwd=work,env=env,check=True)
if __name__=='__main__':main()
