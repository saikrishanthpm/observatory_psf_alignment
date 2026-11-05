"""Forward model of hubbletilus_noise_moves.ipynb for worker processes.

The basis, telescope and PSF functions below are copied unchanged from the notebook so that a
spawned multiprocessing pool can compute the coefficients at the five mirror positions in
parallel (numpy uses Accelerate here, which is not fork-safe). The notebook checks that these
agree with its own genpsf.
"""
import warnings
from functools import lru_cache

import numpy as np
from scipy.special import j1, jv

from optiland import optic, physical_apertures
from optiland.psf import MMDFTPSF

warnings.filterwarnings("ignore")

# Optiland 0.6.3 refuses scalar paraxial analysis (used for the exit pupil of the wavefront reference sphere)
# for decentered or tilted mirrors; 0.6.0, used for the thesis, computed the same approximate values without
# the check. Treat every call as a paraxial seed so that 0.6.3 warns instead of raising; with this, 0.6.3
# reproduces the 0.6.0 Jacobian at the truth to 4e-5.
try:
    import optiland.paraxial_path as _paraxial_path
    _paraxial_path.in_paraxial_seed_scope = lambda: True
except ImportError:  # Optiland 0.6.0 has no such check
    pass

def _is_odd(integer):

    return integer & 1

def noll_indices(j):

    if j < 1:
        raise ValueError("Zernike index j must be a positive integer.")

    # from i, compute m and n
    # I'm not sure if there is an easier/cleaner algorithm or not.
    # This seems semi-complicated to me...

    # figure out which row of the triangle we're in (easy):
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2  # figure out which entry in the row (harder)
        # The rule is that the even Z obtain even indices j, the odd Z odd indices j.
        # Within a given n, lower values of m obtain lower j.

        resid = int(j - nprev - 1)

        if _is_odd(j):
            sign = -1
        else:
            sign = 1

        if _is_odd(n):
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign

    return n, m

def circ_bessel(n, m, npix=128, rho=None, theta=None, outside=np.nan, zero=19.053609871416903, **kwargs):
 
    if theta is None and rho is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / (zero)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                         "provide both of them.")
        
    if not np.all(np.asarray(rho.shape == theta.shape)):
        raise ValueError('The rho and theta arrays do not have consistent shape.')

    if m == 0:
        if n == 0:
            cb_result = j1(2*np.pi*rho) * (1/rho)
        else:
            cb_result = -1**(n/2 + np.abs(m)) * jv(n+1, 2*np.pi*rho) * (1/rho)
    elif m > 0:
        cb_result = -1**(n/2 + np.abs(m)) * jv(n+1, 2*np.pi*rho) * (1/rho) * np.cos(np.abs(m) * theta)
    else:
        cb_result = -1**(n/2 + np.abs(m)) * jv(n+1, 2*np.pi*rho) * (1/rho) * np.sin(np.abs(m) * theta)
    return cb_result


def circ_bessel1(j, **kwargs):

    n, m = noll_indices(j)
    return circ_bessel(n, m, **kwargs)

def basis_vectors(nterms=15, npix=256, rho=None, theta=None, **kwargs):
    
    if rho is not None and theta is not None:
        # both are required, but validated in circbessel1
        shape = rho.shape
        use_polar = True
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                         "provide both of them.")

    else:
        shape = (npix, npix)
        use_polar = False

    basis_output = np.zeros((nterms,) + shape)

    if use_polar:
        for j in range(nterms):
            basis_output[j] = circ_bessel1(j+1, rho=rho, theta=theta, **kwargs)
    else:
        for j in range(nterms):
            basis_output[j] = circ_bessel1(j+1, npix=npix, **kwargs)
    return basis_output



# --- drop-in helpers (uses your existing basis_vectors) ---

def _outside_key(outside):
    if outside is None or (isinstance(outside, float) and np.isnan(outside)):
        return "nan"
    return float(outside)

@lru_cache(maxsize=16)
def _polar_grid(npix, zero):
    x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.0) / zero
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    return rho, theta

@lru_cache(maxsize=16)
def basis_vectors_cached(nterms=15, npix=256, zero=19.053609871416903, outside="nan"):
    """Cached wrapper around basis_vectors. `outside` must be hashable ('nan' or a float)."""
    outside_val = np.nan if outside == "nan" else float(outside)
    rho, theta = _polar_grid(npix, zero)
    return basis_vectors(
        nterms=nterms,
        rho=rho,
        theta=theta,
        outside=outside_val,
        zero=zero,
    )

def clear_basis_cache():
    basis_vectors_cached.cache_clear()
    _polar_grid.cache_clear()

def decompose_psf_cached(psf, nterms=15, outside=np.nan, **kwargs):
    npix = psf.shape[0]
    zero = kwargs.get("zero", 19.053609871416903)
    basis_stack = basis_vectors_cached(
        nterms=nterms,
        npix=npix,
        zero=zero,
        outside=_outside_key(outside),
    )
    basis_flat = basis_stack.reshape(nterms, -1)
    psf_flat = psf.ravel()
    numerators = basis_flat @ psf_flat
    denominators = np.einsum("ij,ij->i", basis_flat, basis_flat)
    return numerators / denominators


def hubble(WL=0.55,sdx=0,sdy=0,sdz=0,stx=0,sty=0):

    K=1000 #mm to micron
    #Hubble
    hubble = optic.Optic(name="Hubble")

    hubble.add_surface(index=0, radius=np.inf, thickness=np.inf)

    # add physical aperture in primary mirror
    obscuration = physical_apertures.RadialAperture(r_max=1200, r_min=384)
    c1 = physical_apertures.OffsetRadialAperture(r_max=75, r_min=0, offset_x=0, offset_y=1000)
    c2 = physical_apertures.OffsetRadialAperture(r_max=75, r_min=0, offset_x=-707.15, offset_y=-707.15)
    c3 = physical_apertures.OffsetRadialAperture(r_max=75, r_min=0, offset_x=707.15, offset_y=-707.15)
    r1 = physical_apertures.RectangularAperture(x_min=-19, x_max=19, y_min=-1200, y_max=1200)
    r2 = physical_apertures.RectangularAperture(x_min=-1200, x_max=1200, y_min=-19, y_max=19)

    ap = obscuration - r1 - r2 -c1 - c2 - c3

    hubble.add_surface(
        index=1,
        x=0.0, y=0.0, z=0,   # paraxial / entrance reference
    )

    hubble.add_surface(
        index=2,
        radius=-11040.02286,
        x=0.0, y=0.0, z=4910.01016,
        material="mirror",
        is_stop=True,
        conic=-1.001152,
        aperture=ap,
    )

    hubble.add_surface(
        index=3,
        radius=-1349.31166,
        x=sdx / K,
        y=sdy / K,
        z=0 + sdz / K,              # independent M2 piston
        rx=np.deg2rad(stx),
        ry=np.deg2rad(sty),
        material="mirror",
        conic=-1.483014,
    )

    hubble.add_surface(
        index=4,
        radius=-635.38227,
        x=0.0, y=0.0, z=6365.20955,         # image fixed while M2 moves
    )

    hubble.set_aperture(aperture_type="EPD", value=2400)

    hubble.set_field_type(field_type="angle")
    hubble.add_field(y=0.)
    hubble.add_field(y=0.038)
    hubble.add_field(y=-0.038)

    hubble.add_wavelength(value=WL, is_primary=True)
    
    return hubble


def genpsf(WL=0.55, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5, field =(0,0)):

    telescope = hubble(WL, sdx, sdy, sdz, stx, sty)
    psf = MMDFTPSF(telescope, field=field, wavelength=WL, grid_size=64)
    PSF = psf.psf[96:160,96:160]
    r = decompose_psf_cached(PSF, nterms=dterms)

    return r


FIELDS = [(0, 0), (0, 1), (1, 0), (1, 1), (-1, -1)]


def position_coeffs(t):
    """Coefficients (5 fields x 15 modes) with the secondary at t = (sdx, sdy, sdz, stx, sty)."""
    return np.array([genpsf(WL=0.55, sdx=t[0], sdy=t[1], sdz=t[2], stx=t[3], sty=t[4], dterms=15, field=f)
                     for f in FIELDS])


def lstep_moves(t, moves, obs):
    """DE cost: squared coefficient residuals summed over the mirror positions t + moves and the fields."""
    t = np.asarray(t, dtype=float)
    return np.sum((obs - np.array([position_coeffs(t + d) for d in moves]))**2)
