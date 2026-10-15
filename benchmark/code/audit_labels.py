import re
from pathlib import Path
import pandas as pd
from common import ROOT,sha,write_json
def main():
    path=Path('E:/octa/OCTAvideo/label.xlsx');original=pd.read_excel(path)
    def normalize(v):return Path(str(v)).stem.strip()
    original['key']=original['AVI文件名'].map(normalize);original['label']=original['是否脑卒'].astype(str).str.strip().map({'是':1,'否':0});assert original.label.notna().all()
    assert original.groupby('key').label.nunique().eq(1).all()
    lookup=original.groupby('key').label.first();manifest=pd.read_csv(ROOT/'audit/all_128_record_manifest.csv');manifest['original_key']=manifest.old_name.map(normalize);manifest['spreadsheet_label']=manifest.original_key.map(lookup)
    assert manifest.spreadsheet_label.notna().all(),'Original spreadsheet lookup failure'
    assert manifest.label.eq(manifest.spreadsheet_label).all(),'Original spreadsheet label conflict'
    manifest[['animal_id','old_name','original_key','label','spreadsheet_label','source_family']].to_csv(ROOT/'audit/original_spreadsheet_label_reconciliation.csv',index=False)
    write_json(ROOT/'audit/label_audit.json',dict(source=str(path),sha256=sha(path),records=128,label_conflicts=0,primary_units=120))
    print('LABEL AUDIT 128 records agree with original spreadsheet; 120 unique source families',flush=True)
if __name__=='__main__':main()
