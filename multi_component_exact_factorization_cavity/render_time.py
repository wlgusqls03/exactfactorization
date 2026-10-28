"""Read-only time selection shared by all cached cavity report renderers."""
from pathlib import Path

import numpy as np


# These are spatial arrays even if their length happens to equal n_times.
SPATIAL_KEYS = frozenset(('R', 'x', 'q', 'Q', 'q_c'))


def validate_t_max(t_max_fs):
    if t_max_fs is not None and (not np.isfinite(t_max_fs) or t_max_fs < 0):
        raise ValueError('t_max_fs must be finite and nonnegative')


def time_mask(times, t_max_fs):
    validate_t_max(t_max_fs)
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError('A finite nonempty 1D time array is required')
    # Only roundoff tolerance: never silently include the next saved frame.
    return np.ones(len(times),dtype=bool) if t_max_fs is None else times<=t_max_fs+1e-10


def select_observables(data, t_max_fs=None):
    """Slice every time-leading observable, including local densities.

    Native observables.npz stores time-leading arrays plus x/R/Q coordinates.
    Do not slice by shape alone: n_grid can coincide with n_times. No array or
    file is modified, and no interpolation or renormalization is performed.
    """
    validate_t_max(t_max_fs)
    if t_max_fs is None or data is None:
        return data
    key='t_fs' if 't_fs' in data else 'times_fs'
    if key not in data: raise ValueError('Cannot apply time limit: observable times unavailable')
    mask=time_mask(data[key],t_max_fs)
    if not mask.any(): raise ValueError('No observables at or before the requested t_max_fs')
    return {k:(np.asarray(v)[mask] if k not in SPATIAL_KEYS and np.ndim(v)>0
               and np.shape(v)[0]==len(mask) else v) for k,v in data.items()}


def select_field_files(files, t_max_fs=None):
    """Read only scalar timestamp members, not large field arrays."""
    validate_t_max(t_max_fs)
    if t_max_fs is None:return files
    selected=[]
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            key=next((k for k in ('t_fs','time_fs') if k in z.files),None)
            if key is None:raise ValueError('Cannot apply time limit: timestamp missing in '+str(path))
            if time_mask([float(z[key])],t_max_fs)[0]:selected.append(path)
    if not selected:raise ValueError('No field frames at or before the requested t_max_fs')
    return selected


def report_directory(case, output_dir=None, t_max_fs=None):
    validate_t_max(t_max_fs)
    if output_dir is not None:return Path(output_dir)
    name='report' if t_max_fs is None else f'report_{float(t_max_fs):g}fs'
    return Path(case)/name


def time_range(data):
    if data is None:return None
    times=data.get('t_fs',data.get('times_fs'))
    if times is None or not len(times):return None
    return [float(np.min(times)),float(np.max(times))]


def constrain_time_axes(fig, t_max_fs=None):
    """Keep matplotlib's automatic margins from extending past the cutoff."""
    if t_max_fs is not None and t_max_fs>0:
        for ax in fig.axes:
            if ax.get_xlabel()=='t (fs)':
                lo,hi=ax.get_xlim()
                upper=min(hi,t_max_fs)
                if upper>max(0.,lo):ax.set_xlim(max(0.,lo),upper)
    return fig
