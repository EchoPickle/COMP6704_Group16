import json
from pathlib import Path
import tempfile
import unittest
import run_experiments as runner

class ProtocolTests(unittest.TestCase):
    def test_selection_uses_validation_not_training_objective(self):
        rows=[dict(config_id=0,validation_macro_f1=.8,validation_accuracy=.9,train_objective=.01),
              dict(config_id=1,validation_macro_f1=.9,validation_accuracy=.8,train_objective=.1)]
        self.assertEqual(runner.select_config(rows),1)

    def test_lock_detects_changed_models(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'m.npz').write_bytes(b'original')
            runner.write_lock(p,{'config':{'lam':.1},'models':['m.npz']})
            self.assertEqual(runner.verify_lock(p)['config']['lam'],.1)
            (p/'m.npz').write_bytes(b'tampered')
            with self.assertRaises(ValueError):runner.verify_lock(p)

    def test_test_phase_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'test_started.json').write_text('{}')
            with self.assertRaises(FileExistsError):runner.evaluate_test(p)

if __name__=='__main__':unittest.main()
