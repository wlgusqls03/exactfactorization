import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .. import phase7_f200_compact_gpu as compact
from ..phase7_f200_compact_gpu import retain_wave,EVENT_STEPS,FINAL_STEP
from ..run_phase7_f200_campaign import RESERVE_BYTES,WAVE_COUNT


class Storage(unittest.TestCase):
    def test_required_times(self):
        for step in EVENT_STEPS|{0,FINAL_STEP}:self.assertTrue(retain_wave(step))
        self.assertFalse(retain_wave(256))
        self.assertTrue(retain_wave(512))

    def test_capacity(self):
        steps=set(range(0,FINAL_STEP+1,512))|EVENT_STEPS|{FINAL_STEP}
        self.assertEqual(len(steps),WAVE_COUNT)
        self.assertLess(RESERVE_BYTES,20_000_000_000)

    def test_storage_only_adapter(self):
        for failed in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                np.savez(root/'observable_0000000.npz',energy=0.)
                np.savez(root/'observable_0000256.npz',energy=0.,norm=2. if failed else 1.,
                         electron_edge=0.,edge=0.,top=0.,continuity_error=0.)
                def fake_main():
                    for name in ('wave_0000256.npz','wave_0000512.npz','restart.npz','observable_0000288.npz'):
                        compact.engine.save_npz(root/name,value=1)
                    return 0
                with patch.object(compact.engine,'save_npz') as save,patch.object(compact.engine,'main',side_effect=fake_main):
                    compact.main()
                    names=[c.args[0].name for c in save.call_args_list]
                    self.assertEqual('wave_0000256.npz' in names,failed)
                    for name in ('wave_0000512.npz','restart.npz','observable_0000288.npz'):self.assertIn(name,names)


if __name__=='__main__':unittest.main()
