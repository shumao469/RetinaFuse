from pathlib import Path
import importlib.util,unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('recompute',ROOT/'scripts/recompute_results.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ResultsTests(unittest.TestCase):
    def test_all_released_metrics(self):self.assertEqual(len(mod.recompute()),43)
    def test_corrected_primary_scope(self):
        q=mod.recompute();r=q[(q.model=='RetinaFuse')&(q.design=='fivefold')&(q.condition=='graychange_4fps_t002')].iloc[0]
        self.assertEqual(int(r.n),120);self.assertAlmostEqual(r.auroc,.975353389,places=8)
    def test_date_failure_retained(self):
        q=mod.recompute();r=q[(q.model=='RetinaFuse')&(q.design=='filename_date')&(q.condition=='graychange_4fps_t002')].iloc[0]
        self.assertEqual(int(r.n),118);self.assertLess(r.auroc,.5)
if __name__=='__main__':unittest.main()
