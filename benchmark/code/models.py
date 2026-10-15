import importlib.util
import torch
from torch import nn
from common import ROOT
spec=importlib.util.spec_from_file_location('company',(ROOT/'vendor/retinafuse_core.py'));company=importlib.util.module_from_spec(spec);spec.loader.exec_module(company)
class ImageBranch(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),nn.Dropout2d(.15),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),nn.Dropout2d(.15),nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),nn.MaxPool2d(2));self.proj=nn.Linear(128*8*8,256);self.out=nn.Dropout(.2)
    def forward(self,x):return self.out(self.proj(self.conv(x).flatten(1)))
class Model(nn.Module):
    def __init__(self,name):
        super().__init__();self.name=name
        if name in ['Single-frame','Mean-image','Unordered-frames']:self.image=ImageBranch()
        elif name in ['Raw-video','Frame-difference-video']:
            self.frame=company.EventFrameBranch();self.frame.conv3d[0]=nn.Conv3d(1,32,3,padding=1)
        else:
            if name!='Statistics-only':self.frame=company.EventFrameBranch()
            if name!='Frame-only':self.stats=company.StatBranch()
            if name=='RetinaFuse':self.gate=company.AdaptiveFusion(256,2)
            elif name=='Global-weight-fusion':self.global_logits=nn.Parameter(torch.zeros(2))
            elif name=='Concat-fusion':self.concat=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(.2))
        self.head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.55),nn.Linear(128,2))
    def forward(self,x,s,gate_mode='learned',clean_weights=None,return_gate=False):
        name=self.name;gate=None
        if name=='Unordered-frames':
            b,t,c,h,w=x.shape;z=self.image(x.reshape(b*t,c,h,w)).reshape(b,t,-1).mean(1)
        elif name in ['Single-frame','Mean-image']:z=self.image(x)
        elif name in ['Raw-video','Frame-difference-video','Frame-only']:z=self.frame(x)
        elif name=='Statistics-only':z=self.stats(s)
        else:
            hf=self.frame(x);hs=self.stats(s)
            if name=='RetinaFuse':
                gate=self.gate.gate(torch.cat([hf,hs],1)).softmax(1)
                if gate_mode=='equal':gate=torch.full_like(gate,.5)
                elif gate_mode=='frozen_clean':gate=clean_weights
            elif name=='Equal-fusion':gate=torch.full((len(x),2),.5,device=x.device)
            elif name=='Global-weight-fusion':gate=self.global_logits.softmax(0).expand(len(x),2)
            if name=='Concat-fusion':z=self.concat(torch.cat([hf,hs],1))
            else:z=gate[:,0,None]*hf+gate[:,1,None]*hs
        logits=self.head(z)
        return (logits,gate)if return_gate else logits
