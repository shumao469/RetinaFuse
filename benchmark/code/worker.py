"""Finite, locked experiment queue with resumable checkpoints and explicit completion evidence."""
from pathlib import Path
import json,os,platform,subprocess,sys,time,traceback
import psutil,torch
from common import ROOT,CFG,sha,write_json,ensure
def queue():
    meta=__import__('pandas').read_csv(ROOT/'data/animal_manifest.csv');jobs=[]
    def add(model,design,held,partition,seed,condition='legacy'):
        name=f'{design}_{held}_{model}_{condition}_seed{seed}';jobs.append(dict(id=name,model=model,design=design,held_out=held,partition_index=partition,seed=seed,condition=condition))
    # Complete seed-42 outer tasks first for all model families, then the two fixed repetitions.
    for seed in CFG['seeds']:
        for fold in range(1,6):
            for model in CFG['models']:add(model,'fivefold',fold,fold,seed)
    for seed in CFG['seeds']:
        for k,day in enumerate(sorted(set(meta.filename_date)-{'unknown'}),6):
            for model in CFG['group_models']:add(model,'filename_date',day,k,seed)
    for seed in CFG['seeds']:
        for condition in CFG['decoder_conditions']:
            for fold in range(1,6):
                for model in CFG['decoder_models']:add(model,'fivefold',fold,fold,seed,condition)
    return jobs
def call(script,args=None,logname=None):
    path=ROOT/'code'/script;dest=ROOT/'logs'/(logname or script.replace('.py','.log'))
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(CFG['gpu']);env['PYTHONUTF8']='1';env['OMP_NUM_THREADS']=str(CFG['cpu_threads']);env['MKL_NUM_THREADS']=str(CFG['cpu_threads'])
    with dest.open('a',encoding='utf-8')as log:
        p=subprocess.Popen([sys.executable,'-u',str(path)]+(args or []),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        write_json(ROOT/'worker_status.json',dict(worker_pid=os.getpid(),child_pid=p.pid,script=script,log=str(dest),updated=time.time(),state='running'))
        return p.wait()
def main():
    ensure();lock=ROOT/'worker.lock';pid=os.getpid()
    if lock.exists():
        old=json.loads(lock.read_text());p=psutil.Process(old['pid'])if psutil.pid_exists(old['pid'])else None
        if p and abs(p.create_time()-old['create_time'])<1:print('Another queue worker is alive',flush=True);return
    write_json(lock,dict(pid=pid,create_time=psutil.Process(pid).create_time()))
    lockfile=ROOT/'audit/environment-lock.txt'
    lockfile.write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    write_json(ROOT/'audit/runtime.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,platform=platform.platform(),gpu_index=CFG['gpu'],gpu_name=torch.cuda.get_device_name(CFG['gpu']),started=time.time()))
    try:
        # Adopt the already-authorized initial preprocessing process instead of duplicating 77 GB of I/O.
        if not(ROOT/'data/prepared.json').exists():
            existing=[p for p in psutil.process_iter(['pid','cmdline','create_time'])if p.pid!=pid and any(str(ROOT/'code/prepare.py').replace('\\','/').lower()==str(a).replace('\\','/').lower()for a in (p.info['cmdline']or []))]
            if existing:
                write_json(ROOT/'worker_status.json',dict(worker_pid=pid,state='waiting_for_existing_input_preparation',adopted_pids=[p.pid for p in existing],updated=time.time()))
                while any(p.is_running()for p in existing):time.sleep(10)
        if not (ROOT/'data/prepared.json').exists():
            if call('prepare.py')!=0:raise RuntimeError('Input preparation failed; inspect prepare.log')
        if call('audit_labels.py')!=0:raise RuntimeError('Original label spreadsheet reconciliation failed')
        if not(ROOT/'audit/preflight_checks.json').exists():
            if call('verify.py')!=0:raise RuntimeError('Preflight forward/backward or lineage failed')
        # Final specification is frozen after input audit, before the first fitting task.
        CFG.clear();CFG.update(json.loads((ROOT/'protocol.json').read_text(encoding='utf-8')))
        current_hashes={p.name:sha(p)for p in (ROOT/'code').glob('*.py')}
        freeze=ROOT/'audit/training_freeze.json'
        spec=dict(protocol_sha256=sha(ROOT/'protocol.json'),manifest_sha256=sha(ROOT/'data/animal_manifest.csv'),implementation=current_hashes)
        if freeze.exists():assert json.loads(freeze.read_text())==spec,'Frozen protocol or implementation changed; do not mix old and new fitting tasks'
        else:write_json(freeze,spec)
        write_json(ROOT/'audit/implementation_hashes.json',current_hashes)
        jobs=queue();write_json(ROOT/'queue.json',jobs)
        locked_hashes=json.loads((ROOT/'audit/implementation_hashes.json').read_text())
        for index,job in enumerate(jobs,1):
            assert all(sha(ROOT/'code'/name)==digest for name,digest in locked_hashes.items()),'Implementation changed after preregistered fitting began'
            complete=ROOT/'runs'/job['id']/'complete.json'
            if complete.exists():
                assert json.loads(complete.read_text())['protocol_sha256']==sha(ROOT/'protocol.json'),'Completed task belongs to another protocol'
                continue
            if job['condition']!='legacy'and not(ROOT/'data/decoder/prepared.json').exists():
                if call('regve.py')!=0:raise RuntimeError('Native video decoder preparation failed')
            print(f'QUEUE {index}/{len(jobs)} {job["id"]}',flush=True)
            write_json(ROOT/'progress.json',dict(total=len(jobs),done=sum((ROOT/'runs'/q['id']/'complete.json').exists()for q in jobs),current=job,state='training',updated=time.time()))
            returncode=call('train.py',['--task',json.dumps(job)],job['id']+'.log')
            if returncode!=0:raise RuntimeError(f'Training failed: {job["id"]}, return code {returncode}')
            if index%11==0:
                if call('analyze.py')!=0:raise RuntimeError('Prediction lineage/analysis failed')
        if call('analyze.py')!=0:raise RuntimeError('Final lineage analysis failed')
        if call('postflight.py')!=0:raise RuntimeError('Final robustness/control verification failed')
        write_json(ROOT/'progress.json',dict(total=len(jobs),done=len(jobs),state='prespecified_local_experiments_complete',updated=time.time(),external_cohort_pending=True))
    except Exception as exc:
        write_json(ROOT/'worker_status.json',dict(worker_pid=pid,state='failed',error=str(exc),traceback=traceback.format_exc(),updated=time.time()));raise
    finally:
        # Retain the lock as an auditable pid/create-time record; restart checks liveness.
        pass
if __name__=='__main__':main()
