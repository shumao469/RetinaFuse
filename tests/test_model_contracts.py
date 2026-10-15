from pathlib import Path
import importlib.util,sys,unittest
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(importlib.util.find_spec('torch') is not None and (ROOT/'benchmark/vendor/retinafuse_core.py').exists(),'Requires compute environment and pinned source bootstrap')
class ModelContracts(unittest.TestCase):
    def test_all_representation_and_fusion_forwards(self):
        sys.path.insert(0,str(ROOT/'benchmark/code'))
        import torch
        from models import Model
        torch.set_num_threads(2)
        names=['Single-frame','Mean-image','Unordered-frames','Raw-video','Frame-difference-video','RetinaFuse','Frame-only','Statistics-only','Equal-fusion','Global-weight-fusion','Concat-fusion']
        with torch.no_grad():
            for name in names:
                model=Model(name).eval();s=torch.zeros(1,12)
                x=torch.zeros(1,1,64,64) if name in ['Single-frame','Mean-image'] else torch.zeros(1,60,1 if name in ['Unordered-frames','Raw-video','Frame-difference-video'] else 2,64,64)
                out,gate=model(x,s,return_gate=True)
                self.assertEqual(tuple(out.shape),(1,2),name);self.assertTrue(torch.isfinite(out).all(),name)
                if gate is not None:self.assertAlmostEqual(float(gate.sum()),1.0,places=6)
if __name__=='__main__':unittest.main()
