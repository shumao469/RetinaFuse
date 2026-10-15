from pathlib import Path
import importlib.util, sys, unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmark/explain'))
from exact_shap import shapley,group_masks,select_donors,coalition_values


class ShapleyContracts(unittest.TestCase):
    def test_additive_exact_solution(self):
        n=12; bits=(np.arange(2**n)[:,None]>>np.arange(n))&1
        beta=np.linspace(-.13,.11,n);values=.3+bits@beta
        np.testing.assert_allclose(shapley(values,n),beta,atol=1e-12)

    def test_pair_interaction_and_additivity(self):
        n=5;bits=(np.arange(2**n)[:,None]>>np.arange(n))&1
        values=.2+.4*bits[:,0]*bits[:,1]
        np.testing.assert_allclose(shapley(values,n),[.2,.2,0,0,0],atol=1e-12)
        self.assertAlmostEqual(values[0]+shapley(values,n).sum(),values[-1])

    def test_group_partition(self):
        np.testing.assert_array_equal(group_masks({'counts':[0,1],'other':[2]},3),[0,3,4,7])
        with self.assertRaises(ValueError):group_masks({'a':[0,1],'b':[1,2]},3)

    def test_invalid_values(self):
        for values,n in [([0,1,2],2),([0,np.nan],1),([0,np.inf],1)]:
            with self.assertRaises(ValueError):shapley(values,n)

    def test_donor_selection_excludes_test_aliases(self):
        data=pd.DataFrame([dict(animal_id='held',label=0,outer_fold=1,source_family='shared'),
                           dict(animal_id='alias',label=0,outer_fold=2,source_family='shared'),
                           dict(animal_id='h',label=0,outer_fold=2,source_family='h'),
                           dict(animal_id='m',label=1,outer_fold=2,source_family='m')])
        q=select_donors(data,1,1,1)
        self.assertEqual(set(q.animal_id),{'h','m'})
        with self.assertRaises(ValueError):select_donors(data,1,2,1)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Requires compute dependencies')
class NetworkCoalitionContracts(unittest.TestCase):
    def test_vectorized_matches_explicit_full_network(self):
        import torch
        torch.manual_seed(42);torch.set_num_threads(1)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.branch_b=torch.nn.Linear(2,2)
                self.fusion=torch.nn.Module();self.fusion.gate=torch.nn.Linear(4,2);self.head=torch.nn.Linear(2,2)
        models=[Model().eval(),Model().eval()]
        hf=[torch.tensor([[.2,-.3]]),torch.tensor([[-.1,.5]])]
        target=torch.tensor([.4,-.2]);background=torch.tensor([[.1,.2],[-.2,.3],[.7,-.5]])
        values=coalition_values(models,hf,target,background,batch_size=2)
        manual=[]
        with torch.no_grad():
            for mask in range(4):
                donor_probs=[]
                for donor in background:
                    x=torch.tensor([target[j]if mask&(1<<j)else donor[j]for j in range(2)])[None]
                    margins=[]
                    for net,ff in zip(models,hf):
                        hs=net.branch_b(x);alpha=net.fusion.gate(torch.cat([ff,hs],-1)).softmax(-1)
                        logits=net.head(alpha[:,0,None]*ff+alpha[:,1,None]*hs)
                        margins.append(logits[0,1]-logits[0,0])
                    donor_probs.append(torch.sigmoid(torch.stack(margins).mean()).item())
                manual.append(np.mean(donor_probs))
        np.testing.assert_allclose(values,manual,atol=1e-7)
        self.assertAlmostEqual(float(values[0]+shapley(values,2).sum()),float(values[-1]),places=12)


if __name__=='__main__':unittest.main()
