"""Only remove explicitly named transfer archives with byte-identical extracts.

Never deletes inputs, wavefunctions, restart files, logs or diagnostic folders.
Default is a dry run. Run on an idle server repository, not during extraction.
"""
import argparse
import hashlib
from pathlib import Path
import tarfile


def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.digest()


def verified(archive,target):
    count=0
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            name=Path(member.name)
            if name.is_absolute() or '..' in name.parts:return False,'unsafe member'
            if member.isdir():continue
            if not member.isfile():return False,'non-regular member'
            path=target/name
            if not path.resolve().is_relative_to(target.resolve()):return False,'outside target'
            if not path.is_file() or path.stat().st_size!=member.size:return False,'missing/different '+str(path)
            with path.open('rb') as local,tar.extractfile(member) as packed:
                if digest(local)!=digest(packed):return False,'content mismatch '+str(path)
            count+=1
    return count>0,str(count)+' verified files'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delete-verified-archives',action='store_true')
    a=p.parse_args();repo=Path.cwd();base=repo/'results/vsc_polariton/phase6_gpu'
    if not (repo/'multi_component_exact_factorization/core.py').is_file():
        p.error('Run from exactfactorization repository root')
    targets={'phase6_gpu_server_bundle.tar.gz':base/'transfer',
             'phase6_completion_server_bundle.tar.gz':base/'completion_transfer',
             'phase6_free_recovery_server_bundle.tar.gz':base/'free_recovery_transfer_v2',
             'phase6_finish_server_bundle.tar.gz':base/'finish_transfer_v1',
             'phase6_free_recovery_results.tar.gz':base,
             'phase6_finish_results.tar.gz':base,
             'phase6_finish_v2_results.tar.gz':base,
             'phase6_finish_v3_results.tar.gz':base}
    total=0
    for name,target in targets.items():
        for folder in (repo,base):
            archive=folder/name
            if not archive.is_file() or archive.is_symlink():continue
            try:ok,why=verified(archive,target)
            except (OSError,tarfile.TarError) as exc:ok,why=False,str(exc)
            if not ok:
                print('KEEP',archive,why);continue
            size=archive.stat().st_size;total+=size
            print('DELETE' if a.delete_verified_archives else 'SAFE_DUPLICATE',archive,size,why)
            if a.delete_verified_archives:archive.unlink()
    print(('Freed' if a.delete_verified_archives else 'Reclaimable'),f'{total/1024**3:.3f} GiB')
    print('Extracted inputs, waves, all successful/failed diagnostics remain untouched.')


if __name__=='__main__':main()
