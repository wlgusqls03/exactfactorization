"""Fetch pinned public sources, add explicit I/O instrumentation, build locally.

No sudo, pip installation, unpinned dependency downloads or in-place edits to an
existing source checkout. See README.txt for the LGPL notices and limitations.
"""
import argparse
import ctypes.util
import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
PINS = {
    'gctmqc': ('b7a2a6d95656e4e0535e744b102670d85b07fb88',
              'https://gitlab.com/agostini.work/g-ctmqc/-/archive/{sha}/g-ctmqc-{sha}.tar.gz'),
    'QuantumModelLib': ('64d14d724dc5c5fc6350c837d93c3b2c35bca4ee',
                       'https://codeload.github.com/lauvergn/QuantumModelLib/tar.gz/{sha}'),
    'AD_dnSVM': ('32db0bd122ce1dae290e9cf4c1fda4ab8f3de6f1',
                'https://codeload.github.com/lauvergn/AD_dnSVM/tar.gz/{sha}'),
    'QDUtilLib': ('8a1fa9ca7cf52683c98cc4bcb40f21dc41df6ed5',
                 'https://codeload.github.com/lauvergn/QDUtilLib/tar.gz/{sha}'),
}
ARCHIVE_SHA256 = {
    'gctmqc':'962e5441b03da685d5255804c312abec6a0e4ea6d0cf0f257ab0cb0a6c377597',
    'QuantumModelLib':'4bcb586716fe31644e66a5d3101da069eefc4406caa8ba7525bf6e1bdd4f1854',
    'AD_dnSVM':'021f923a7f7d01be51416456b96bcfe4fb3ae3207d04fe0d71d03208b570c554',
    'QDUtilLib':'797b427653a60b8cf760c13400c35fc7b79fad59915b1fa508eac97de207281b',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(data, target):
    """Strip one archive root; reject escapes and links before any extraction."""
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
        members = tar.getmembers()
        roots = {m.name.split('/')[0] for m in members}
        if len(roots) != 1:
            raise ValueError('Expected one archive root')
        # These two kinds of links are recreated by our build/its makefile.
        skip_links = {'Ext_Lib/QuantumModelLib', 'Ext_Lib/AD_dnSVM',
                      'Ext_Lib/QDUtilLib', 'SRC/QML/ExtModel_m.f90'}
        for m in members:
            rel = Path(*Path(m.name).parts[1:])
            if m.issym() and rel.as_posix() in skip_links:
                continue
            if '..' in rel.parts or rel.is_absolute() or not (m.isfile() or m.isdir()):
                raise ValueError('Unsafe archive member: '+m.name)
        for m in members:
            if m.issym():
                continue
            rel = Path(*Path(m.name).parts[1:])
            # Parent archives can bundle old *_loc dependency trees. Do NOT
            # overlay newer archives on top: deleted old sources could survive.
            # Each dependency is instead extracted exactly once from its pin.
            if len(rel.parts)>1 and rel.parts[0]=='Ext_Lib' and rel.parts[1] not in ('.gitignore','makefile-extlib'):
                continue
            p = target/rel
            if m.isdir():
                p.mkdir(parents=True, exist_ok=True)
            elif rel.parts:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(tar.extractfile(m).read())
                p.chmod(m.mode & 0o777)


def instrument(source):
    """Only initialization/I/O/norm observation; no changes to equations/steps.

    Every replacement must occur exactly once in the pinned source. Save the
    actual unified diff alongside the build for independent inspection.
    """
    edits = {
        'variables.f90': [
            ('  namelist /system/', '  real(kind=dp) :: adapter_norm_max = 0.0_dp\n\n  namelist /system/'),
        ],
        'coefficients_evolution.f90': [
            ('    coeff = coeff/sqrt(normalization)',
             '    adapter_norm_max = max(adapter_norm_max,abs(normalization-1.0_dp))\n'
             '    coeff = coeff/sqrt(normalization)'),
        ],
        'time_evolution.f90': [
            ('    CALL plot_STC(k_li,BOsigma,my_force_E,Vcl,0)',
             '    CALL plot_STC(k_li,BOsigma,my_force_E,Vcl,0)\n    CALL adapter_snapshot(0)'),
            ('    !CALL plot_QMOM(Rcl,qmom,qmom_type,time)',
             '    IF(mod(time,dump)==0 .OR. time==nsteps) CALL adapter_snapshot(time)\n'
             '    !CALL plot_QMOM(Rcl,qmom,qmom_type,time)'),
            ('    IF (.NOT. new_potential) THEN\n      IF (.NOT. ALLOCATED(Vec0)) THEN',
             '    CALL adapter_initial_coefficients\n\n'
             '    IF (.NOT. new_potential) THEN\n      IF (.NOT. ALLOCATED(Vec0)) THEN'),
            ('END MODULE time_evolution',
             "INCLUDE 'adapter_io.inc'\nEND MODULE time_evolution"),
        ],
    }
    diff = []
    for filename, replacements in edits.items():
        path = source/'src'/filename
        old = path.read_text()
        new = old
        for before, after in replacements:
            if new.count(before) != 1:
                raise ValueError('Pinned source mismatch: '+filename+' '+repr(before))
            new = new.replace(before, after)
        path.write_text(new)
        diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                    fromfile='upstream/src/'+filename, tofile='adapter/src/'+filename))
    shutil.copyfile(HERE/'adapter_io.inc', source/'src'/'adapter_io.inc')
    return ''.join(diff)


def build(out, download=False, retry=False, archive_cache=None):
    out = Path(out).resolve()
    if (out/'build_manifest.json').exists() and not retry:
        m = json.loads((out/'build_manifest.json').read_text())
        if m['binary_sha256'] != digest(out/'source/src/main.x'):
            raise ValueError('Executable changed since build; use a fresh build directory')
        if m['probe_sha256'] != digest(out/'model_probe.x') or any(
                m['adapter_sha256'][f]!=digest(HERE/f) for f in m['adapter_sha256']):
            raise ValueError('Adapter changed: use --retry-build to verify/rebuild')
        print('REUSE', out/'source/src/main.x', flush=True)
        return m
    for exe in ('gfortran', 'make'):
        if not shutil.which(exe):
            raise RuntimeError(exe+' unavailable; install on the server, not via this script')
    out.mkdir(parents=True, exist_ok=True)
    # Archives are reusable after interrupted downloads/builds. Source trees are
    # never overwritten automatically: retain failed builds for diagnosis.
    existed = (out/'source').exists()
    if existed and not retry:
        raise FileExistsError('Incomplete build exists; choose a NEW --out (archives may be copied)')
    archives = Path(archive_cache).resolve() if archive_cache else out/'archives'
    archives.mkdir(exist_ok=True)
    qml = out/'source/Ext_Lib/QuantumModelLib_loc'
    targets = {'gctmqc': out/'source', 'QuantumModelLib': qml,
               'AD_dnSVM': qml/'Ext_Lib/AD_dnSVM_loc', 'QDUtilLib': qml/'Ext_Lib/QDUtilLib_loc'}
    hashes = {}
    for name, (sha, url) in PINS.items():
        archive = archives/(name+'-'+sha+'.tar.gz')
        if not archive.exists():
            if not download:
                raise FileNotFoundError(str(archive)+'; use --download to explicitly allow retrieval')
            print('DOWNLOAD', name, sha, flush=True)
            data = urllib.request.urlopen(url.format(sha=sha), timeout=60).read()
            # Validate fully before caching (e.g. don't save an HTTP error page).
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
                tar.getmembers()
            archive.write_bytes(data)
        hashes[name] = dict(commit=sha, archive_sha256=digest(archive), url=url.format(sha=sha))
        if hashes[name]['archive_sha256'] != ARCHIVE_SHA256[name]:
            raise ValueError('Archive checksum mismatch: '+str(archive))
        if not existed:
            unpack(archive.read_bytes(), targets[name])
        else:
            # A retry may reuse object files, but must not reuse changed physics.
            with tempfile.TemporaryDirectory(prefix='gctmqc-verify-') as tmp:
                fresh=Path(tmp)
                unpack(archive.read_bytes(),fresh)
                if name=='gctmqc':
                    instrument(fresh)
                for f in fresh.rglob('*'):
                    if 'Ext_Lib' in f.relative_to(fresh).parts:
                        continue  # independently pinned/checked below
                    if f.is_file() and f.suffix.lower() in ('.f90','.f','.inc'):
                        other=targets[name]/f.relative_to(fresh)
                        if not other.exists() or digest(f)!=digest(other):
                            raise ValueError('Changed source on retry: '+str(other))
    if not existed:
        for parent, name in ((out/'source/Ext_Lib', 'QuantumModelLib'),
                             (qml/'Ext_Lib', 'AD_dnSVM'), (qml/'Ext_Lib', 'QDUtilLib')):
            (parent/name).symlink_to(name+'_loc', target_is_directory=True)
        (out/'adapter.patch').write_text(instrument(out/'source'))
    suffix = '_gfortran_opt1_omp1_lapack1_int4_real64.a'
    lib = qml/('libQuantumModelLibFull'+suffix)
    # WITHRK16=0 concerns unused library quadruple-precision routines only.
    # G-CTMQC's own qp coefficients are unchanged. Build one job: dependency
    # makefiles in this version can race when invoked recursively in parallel.
    # Debian runtime-only installs expose .so.3 but no development .so link.
    # Link the system ABI explicitly; never borrow NumPy's private BLAS ABI.
    links=[]
    for name in ('lapack','blas'):
        libname=ctypes.util.find_library(name)
        if not libname:
            raise RuntimeError('System '+name+' unavailable; see server prerequisites')
        links.append('-l:'+libname if '.so' in libname else '-l'+name)
    commands = [
        ['make', '-C', str(qml), 'lib', '-j1', 'WITHRK16=0'],
        ['make', '-C', str(out/'source/src'), '-j1',
         'ExtLibDIR='+str(out/'source/Ext_Lib'), 'QMLLib='+str(lib),'LFLAG='+' '.join(links)],
    ]
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (out/'build.log').open('w') as log:
        for command in commands:
            print('BUILD', ' '.join(command), flush=True)
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        # A tiny QML-only executable checks the actual library matrix, gradient,
        # and eigenvector conventions independently of propagation.
        command = ['gfortran', '-O2', '-fopenmp', str(HERE/'model_probe.f90'),
                   str(lib), *links, '-o', str(out/'model_probe.x')]
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    manifest = dict(backend='authors-gctmqc-instrumented', sources=hashes,
        adapter_sha256={f:digest(HERE/f) for f in ('build.py','adapter_io.inc','model_probe.f90')},
        modifications='External initial coefficients, binary observations and pre-normalization norm monitor ONLY',
        paper_exact_commit=False, compiler=subprocess.check_output(['gfortran','--version'], text=True).splitlines()[0],
        commands=commands, binary_sha256=digest(out/'source/src/main.x'),
        probe_sha256=digest(out/'model_probe.x'))
    (out/'build_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('BUILT', out/'source/src/main.x', flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    p.add_argument('--retry-build',action='store_true',help='Verify pinned sources and retry an interrupted build')
    p.add_argument('--archive-cache',type=Path,help='Reuse downloaded pinned archives without network')
    a = p.parse_args()
    build(a.out, a.download,a.retry_build,a.archive_cache)


if __name__ == '__main__':
    main()
