"""Finish remaining controls with spectral point flux; preserve v1/v2 data."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
from . import run_phase6_finish as finish
from .run_phase6_gpu import save_json


def main():
    base=Path('results/vsc_polariton/phase6_gpu')
    prior=base/'finish_campaign_v2'
    stationary_file=prior/'controls/stationary.json'
    source_manifest=json.loads((prior/'finish_baseline.json').read_text())
    # The already-passed stationary calculation used exactly these sources and
    # the selected packet; never reuse just an unbound PASS flag.
    for name in ('phase6_stationary_compat.py','phase6_gpu_backend.py','run_phase6_gpu.py'):
        path=Path('multi_component_exact_factorization/vsc_polariton')/name
        finish.verify_manifest({str(path):source_manifest['sources'][str(path)]})
    finish.verify_manifest(source_manifest['inputs'])
    selection=base/'free_recovery_campaign_v1/free_selection.json'
    finish.verify_manifest({str(selection):source_manifest['old_outputs'][str(selection)]})
    s=json.loads(stationary_file.read_text())
    if not (s['status']=='PASS' and s['eigen_residual']<1e-9 and s['density_L1']<1e-6
            and abs(s['final_norm']-1)<1e-9 and abs(s['energy_drift'])<1e-6):
        raise RuntimeError('Stationary reuse criteria failed')
    def reuse(packet,out,device):
        digest=hashlib.sha256(stationary_file.read_bytes()).hexdigest()
        record=dict(s,reused_from=str(stationary_file),source_sha256=digest)
        if out.exists():
            if json.loads(out.read_text())!=record:raise RuntimeError('Stationary record changed')
        else:save_json(out,record)
    original_run=finish.run
    def run(command,log):
        command=list(command)
        old=finish.PREFIX+'phase6_completion_controls'
        if old in command:command[command.index(old)]=finish.PREFIX+'phase6_control_flux'
        if finish.PREFIX+'tests.test_phase6_finish' in command:
            command.append(finish.PREFIX+'tests.test_phase6_control_flux')
        return original_run(command,log)
    argv=list(sys.argv)
    if '--out' not in argv:argv+=['--out',str(base/'finish_campaign_v3')]
    with patch.object(finish,'stationary',side_effect=reuse),patch.object(finish,'run',side_effect=run),patch.object(sys,'argv',argv):
        finish.main()


if __name__=='__main__':main()
