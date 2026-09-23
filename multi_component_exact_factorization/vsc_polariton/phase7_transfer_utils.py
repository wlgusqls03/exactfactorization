"""Portable file/event helpers: no plotting or local audit dependencies."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

AU_FS=0.024188843265857


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def events(time,product,flux):
    """Same saved-time event rule used by the reaction media renderer."""
    forward=find_peaks(flux)[0];back=find_peaks(-flux)[0]
    forward=forward[flux[forward]>0];back=back[flux[back]<0]
    forward=forward[flux[forward]>.01*np.max(flux)] if len(forward) else forward
    back=back[-flux[back]>.01*np.max(-flux)] if len(back) else back
    return {'initial':0,'first_forward_peak':int(forward[0]) if len(forward) else None,
            'first_backward_peak':int(back[0]) if len(back) else None,
            'max_product':int(np.argmax(product)),'final':len(time)-1}
