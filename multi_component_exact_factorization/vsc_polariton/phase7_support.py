"""Probability-budget masks; no physical values are filled across nodes."""
import numpy as np
from scipy.ndimage import label


def budget_support(density, weights, budget):
    if not 0<budget<1:raise ValueError('Invalid excluded probability budget')
    density=np.asarray(density)
    weights=np.broadcast_to(weights,density.shape)
    if not np.isfinite(density).all() or np.any(density<0):raise ValueError('Invalid density')
    mass=(density*weights).ravel();total=float(mass.sum())
    if total<=0:raise ValueError('Empty density')
    order=np.argsort(-density.ravel(),kind='stable')
    end=min(np.searchsorted(np.cumsum(mass[order]),(1-budget)*total),len(order)-1)
    threshold=density.ravel()[order[end]]
    mask=(density>=threshold)&(density>0)
    components,count=label(mask)
    return mask,dict(threshold=float(threshold),excluded=float(mass[~mask.ravel()].sum()/total),
                     components=int(count)),components


def weighted_rms(value, mass, mask):
    valid=mask&(mass>0)
    if not valid.any():return None
    if not np.isfinite(value[valid]).all():return float('inf')
    return float(np.sqrt(np.sum(mass[valid]*abs(value[valid])**2)/np.sum(mass[valid])))
