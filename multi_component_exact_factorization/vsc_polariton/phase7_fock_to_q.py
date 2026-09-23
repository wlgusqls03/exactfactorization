"""Physical Hermite functions and Gauss--Hermite quadrature (atomic units).

Input coefficients (...,n); output (...,q). Derivatives retain n=N,N+1
headroom rather than squaring a truncated ladder matrix.
"""
import numpy as np
from scipy.special import roots_hermite


def hermite_grid(nf, nq, omega):
    if nf<1 or nq<nf+2 or omega<=0:raise ValueError('Require nq>=nf+2, omega>0')
    z,w=roots_hermite(nq)
    q=z/np.sqrt(omega)
    weights=np.exp(np.log(w)+z*z)/np.sqrt(omega)
    basis=np.empty((nf+2,nq))
    basis[0]=(omega/np.pi)**.25*np.exp(-z*z/2)
    basis[1]=np.sqrt(2)*z*basis[0]
    for n in range(1,nf+1):
        basis[n+1]=np.sqrt(2/(n+1))*z*basis[n]-np.sqrt(n/(n+1))*basis[n-1]
    d=np.empty((nf,nq));dd=np.empty_like(d)
    for n in range(nf):
        lower=np.sqrt(n)*basis[n-1] if n else 0.
        lower2=np.sqrt(n*(n-1))*basis[n-2] if n>1 else 0.
        d[n]=np.sqrt(omega/2)*(lower-np.sqrt(n+1)*basis[n+1])
        dd[n]=omega/2*(lower2-(2*n+1)*basis[n]+np.sqrt((n+1)*(n+2))*basis[n+2])
    b=basis[:nf]
    error=float(np.max(abs((b*weights)@b.T-np.eye(nf))))
    return dict(q=q,weights=weights,basis=b,d_basis=d,dd_basis=dd,orthogonality=error)


def transform(coefficients, grid, derivative=0):
    key={0:'basis',1:'d_basis',2:'dd_basis'}[derivative]
    return coefficients@grid[key]


def backproject(values,grid):
    return (values*grid['weights'])@grid['basis'].T
