"""Local-only exact eta=0 vacuum-sector packets; historical code read-only.

Predeclared boxes 28.8,36,48,72 a0; dx .3 and .24; R grid unchanged.
No electronic BO truncation in propagation. NF=1 is an exact invariant
sector ONLY because eta=0 and the initial photon is the vacuum.
"""
from . import phase6_resume as resume
from .phase6_gpu_export import export

BOXES = (28.8, 36., 48., 72.)


def main():
    root = resume.ROOT/'results/vsc_polariton/phase6_gpu/free_recovery_transfer'
    previous = resume.OUT
    try:
        resume.OUT = root/'local_build'
        for half in BOXES:
            for dx in (.3, .24):
                n = 2*round(half/dx)
                name = f'free_L{half:g}_dx{dx:g}'
                path = root/'inputs'/(name+'.npz')
                if path.exists():
                    raise FileExistsError('Preserve existing packet: '+str(path))
                if name in resume.SETTINGS:
                    raise RuntimeError('Preset collision')
                resume.SETTINGS[name] = (half,n,4.4,352,1,.125)
                try:
                    export(path,'free',name)
                finally:
                    del resume.SETTINGS[name]
    finally:
        resume.OUT = previous


if __name__=='__main__':main()
