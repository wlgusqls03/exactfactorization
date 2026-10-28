"""Magi et al., PCCP (2026), DOI 10.1039/D6CP02020H, Eqs. 21--27.

Native wave axes: [R, electronic diabatic label, photon Fock number].
Default: 500 periodic nuclear points [-2,10), two electronic and TWO photon
states (n=0,1). This reproduces the paper's model, NOT its 500-point photon
grid quantum reference. DSE-on and weak coupling are additional controls.
No rotating-wave approximation. All quantities are in atomic units.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.special import eval_hermite, gammaln

AU_FS = 0.024188843265857


@dataclass(frozen=True)
class Config:
    nr: int = 500
    rmin: float = -2.
    rmax: float = 10.
    dt: float = .1
    mass: float = 20000.
    omega: float = .17
    g: float = .01
    dse: bool = False
    nf: int = 2
    k: float = .02
    a: float = 3.
    b: float = .01
    r1: float = 6.
    r2: float = 2.
    r3: float = 3.875
    delta: float = 0.
    z: float = 1.
    mu12: float = -1.

    def grid(self):
        numeric = [v for v in asdict(self).values() if not isinstance(v, bool)]
        if not np.all(np.isfinite(numeric)):
            raise ValueError('Non-finite configuration')
        if self.nr < 8 or self.nf < 2 or self.nr != int(self.nr) or self.nf != int(self.nf):
            raise ValueError('Need integer nr>=8 and nf>=2')
        if min(self.dt, self.mass, self.omega, self.k) <= 0 or self.rmax <= self.rmin:
            raise ValueError('Invalid grid, mass, frequency or time step')
        if not isinstance(self.dse, bool):
            raise ValueError('dse must be a JSON boolean')
        return np.linspace(self.rmin, self.rmax, self.nr, endpoint=False)

    @property
    def sigma_r(self):
        # Eq.25 amplitude width; density standard deviation is sigma_r/sqrt(2).
        return (self.mass*self.k)**(-.25)


def derivative(u, dr, order=1):
    k = 2*np.pi*np.fft.fftfreq(len(u), dr)
    return np.fft.ifft(np.fft.fft(u, axis=0)*(1j*k.reshape((-1,)+(1,)*(u.ndim-1)))**order, axis=0)


def interval_integral(values, R, lower, upper):
    """Integrate the native trigonometric interpolant, including partial cells.

    A simple sum over R>=4 includes a whole cell at the dividing point and
    creates an O(dR) apparent grid error. This integral avoids that bias.
    """
    dr = R[1]-R[0]
    k = 2*np.pi*np.fft.fftfreq(len(R), dr)
    coeff = np.fft.fft(values)/len(R)
    weights = np.empty(len(R), dtype=complex)
    weights[0] = upper-lower
    weights[1:] = (np.exp(1j*k[1:]*(upper-R[0]))-np.exp(1j*k[1:]*(lower-R[0])))/(1j*k[1:])
    return float(np.sum(coeff*weights).real)


def electronic(c, R):
    """H_BO and model dipole D=ZR I + mu12 sigma_x in the diabatic basis."""
    R = np.asarray(R)
    h = np.zeros((len(R), 2, 2))
    h[:, 0, 0] = .5*c.k*(R-c.r1)**2
    h[:, 1, 1] = .5*c.k*(R-c.r2)**2+c.delta
    h[:, 0, 1] = h[:, 1, 0] = c.b*np.exp(-c.a*(R-c.r3)**2)
    dip = c.z*R[:, None, None]*np.eye(2)
    dip[:, 0, 1] = dip[:, 1, 0] = c.mu12
    return h, dip


def photon_operators(c):
    """Project full oscillator operators BEFORE multiplication/truncation.

    P q^2 P != (P q P)^2 at the top Fock state; same for p^2.
    This distinction is essential for N = omega*q^2/2+p^2/(2*omega)-1/2.
    """
    a = np.diag(np.sqrt(np.arange(1, c.nf+1)), 1)
    q = (a+a.T)/np.sqrt(2*c.omega)
    p = 1j*np.sqrt(c.omega/2)*(a.T-a)
    cut = np.s_[:c.nf, :c.nf]
    return dict(q=q[cut], p=p[cut], q2=(q@q)[cut], p2=(p@p)[cut],
                number=np.diag(np.arange(c.nf)), energy=np.diag(c.omega*(np.arange(c.nf)+.5)))


def local_hamiltonian(c, R):
    """H-T_R in a 2*nf basis; electronic-major order e*nf+n.

    DSE=.5*g^2*D@D is the self-polarization of THIS defined two-state dipole.
    It is not claimed to equal projection of an unknown full-electron D^2.
    """
    h, dip = electronic(c, R)
    ops = photon_operators(c)
    dse = .5*c.g**2*(dip@dip) if c.dse else np.zeros_like(dip)
    kron = lambda x, y: np.einsum('rij,nm->rinjm', x, y).reshape(len(R), 2*c.nf, 2*c.nf)
    v = kron(h+dse, np.eye(c.nf))+np.kron(np.eye(2), ops['energy'])[None]
    v += c.omega*c.g*kron(dip, ops['q'])
    return v


def initial(c):
    """Eq.27: bare adiabatic S1, real Gaussian at R2, exact Fock vacuum."""
    R = c.grid()
    _, states = np.linalg.eigh(electronic(c, R)[0])
    s1 = states[:, :, 1].copy()
    for j in range(1, len(R)):
        if np.dot(s1[j-1], s1[j]) < 0:
            s1[j] *= -1
    u = np.zeros((len(R), 2, c.nf), dtype=complex)
    u[:, :, 0] = np.exp(-.5*((R-c.r2)/c.sigma_r)**2)[:, None]*s1
    u /= np.sqrt(np.sum(abs(u)**2)*(R[1]-R[0]))
    return u


def oscillator(c, q):
    """Analytic HO functions and first/second q derivatives, no FFT at q edges."""
    q = np.asarray(q)
    z = np.sqrt(c.omega)*q
    h = np.array([(c.omega/np.pi)**.25*np.exp(-z*z/2-.5*(n*np.log(2)+gammaln(n+1)))
                  *eval_hermite(n, z) for n in range(c.nf+1)])
    dh = np.array([np.sqrt(c.omega*n/2)*h[n-1] if n else np.zeros_like(q)
                   for n in range(c.nf)])
    dh -= np.sqrt(c.omega*np.arange(1, c.nf+1)/2)[:, None]*h[1:]
    ddh = (c.omega**2*q[None]**2-c.omega*(2*np.arange(c.nf)[:, None]+1))*h[:c.nf]
    return h[:c.nf], dh, ddh


def cbo(c, R, q):
    """Static unprojected local-q electronic references, not finite-Fock PESs."""
    h, dip = electronic(c, R)
    v = h[:, None]+c.omega*c.g*np.asarray(q)[None, :, None, None]*dip[:, None]
    v += .5*c.omega**2*np.asarray(q)[None, :, None, None]**2*np.eye(2)
    if c.dse:
        v += .5*c.g**2*(dip@dip)[:, None]
    return np.linalg.eigh(v)


class Propagator:
    """Static local 4x4 exponential cached once; only nuclear FFT each step."""
    def __init__(self, c):
        self.c = c
        self.R = c.grid()
        self.dr = self.R[1]-self.R[0]
        self.v = local_hamiltonian(c, self.R)
        self.pol_energy, self.pol_vectors = np.linalg.eigh(self.v)
        e, v = self.pol_energy, self.pol_vectors
        self.half = (v*np.exp(-.5j*c.dt*e)[:, None, :])@v.transpose(0, 2, 1)
        k = 2*np.pi*np.fft.fftfreq(c.nr, self.dr)
        self.kinetic = np.exp(-.5j*c.dt*k*k/c.mass)[:, None]
        self.bo_energy, self.bo_vectors = np.linalg.eigh(electronic(c, self.R)[0])
        self.ops = photon_operators(c)

    def step(self, u):
        flat = u.reshape(self.c.nr, -1)
        flat = np.einsum('rij,rj->ri', self.half, flat)
        flat = np.fft.ifft(np.fft.fft(flat, axis=0)*self.kinetic, axis=0)
        return np.einsum('rij,rj->ri', self.half, flat).reshape(u.shape)

    def action(self, u):
        return (-derivative(u, self.dr, 2)/(2*self.c.mass)
                +np.einsum('rij,rj->ri', self.v, u.reshape(self.c.nr, -1)).reshape(u.shape))


def observables(u, p):
    c, dr, R = p.c, p.dr, p.R
    density = np.sum(abs(u)**2, axis=(1, 2))
    norm = density.sum()*dr
    expectation = lambda op: float(np.einsum('ren,nm,rem->', u.conj(), op, u).real*dr/norm)
    bo = np.einsum('rek,ren->rkn', p.bo_vectors, u)
    pol = np.einsum('rik,ri->rk', p.pol_vectors, u.reshape(c.nr, -1))
    ur = derivative(u, dr)
    j = np.sum((u.conj()*ur).imag, axis=(1, 2))/c.mass
    # Coupling to the first omitted oscillator state; norm of Q H P Psi.
    dip = electronic(c, R)[1]
    leak = c.g*np.sqrt(c.omega*c.nf/2)*np.einsum('rij,rj->ri', dip, u[:, :, -1])
    return dict(norm=float(norm), energy=float(np.vdot(u, p.action(u)).real*dr/norm),
                kinetic_R=float(np.sum(abs(ur)**2)*dr/(2*c.mass*norm)),
                photon_number=expectation(p.ops['number']), photon_energy=expectation(p.ops['energy']),
                q_mean=expectation(p.ops['q']), p_mean=expectation(p.ops['p']),
                q2=expectation(p.ops['q2']), p2=expectation(p.ops['p2']),
                bo_population=(np.sum(abs(bo)**2, axis=(0, 2))*dr/norm).tolist(),
                pol_population=(np.sum(abs(pol)**2, axis=0)*dr/norm).tolist(),
                top_fock_population=float(np.sum(abs(u[:, :, -1])**2)*dr/norm),
                omitted_coupling_norm_Ha=float(np.sqrt(np.sum(abs(leak)**2)*dr/norm)),
                product_R_ge_4=interval_integral(density, R, 4., c.rmax)/norm,
                signed_flux_R4=float(np.interp(4., R, j)/norm),
                edge_probability=float(density[(R < c.rmin+.3)|(R > c.rmax-.3)].sum()*dr/norm))
