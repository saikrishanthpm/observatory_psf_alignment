import os

os.environ['OMP_NUM_THREADS'] = '4'  # Set the number of threads for parallel processing

import numpy as np
import scipy
from Batoid4LOFT import *
import dynesty
from scipy.special import jv
from scipy.special import j1
from scipy.optimize import minimize
from multiprocessing import Pool
from functools import lru_cache

def telescopeCreation(WL,pdx,pdy,pdz,ptx,pty,sdx,sdy,sdz,stx,sty):
    #gaberration inputs in microns and arcseconds
    global telescope, M1

    global outter1, inner1, outterD
    
    K = 1000000 #meter to micron conversion
    A = 3600 #degree to arcsec conversion
    
    R1, conic1, inner1, outter1 = -16.256, -0.9953569, 1.38, 6.42
    #M1coord = defineCoordinate(anglX=0.0035, anglY=0.0035, anglZ=0, posX=200e-6, posY=200e-6, posZ=200e-6).local()
    M1coord = defineCoordinate(posX=pdx/K, posY=pdy/K, posZ=pdz/K, anglX=ptx/A, anglY=pty/A).local()
    M1 = optic('M1',type='Quadric', RoC=R1, conic=conic1, inDiam=inner1, outDiam=outter1, coordSys=M1coord).mirrorWithSpider(width=0.3, height=4)
                                                                                                                             
    R2, conic2, outter2 = -1.660575, -1.566503, 0.7147252
    #M2coord = defineCoordinate(anglX=-0.00878, anglY=0.0116, anglZ=0, posX=-0.2014e-3, posY=0.8984e-3, posZ=-7.400+191.3e-6).local()
    M2coord = defineCoordinate(posX=sdx/K, posY=sdy/K, posZ=-7.400+(sdz/K), anglX=stx/A, anglY=sty/A).local()
    M2 = optic('M2', type='Quadric', RoC=R2, conic=conic2, outDiam=outter2, coordSys=M2coord).mirror()

    R3, conic3, outter3 = -1.830517, -0.718069, 0.920733
    M3coord = defineCoordinate(posZ=+0.050).local()
    M3 = optic('M3', type='Quadric', RoC=R3, conic=conic3, outDiam=outter3, coordSys=M3coord).mirror()

    outter4 = 0.0991468
    M4coord = defineCoordinate(posZ=-0.983629).local()
    M4 = optic('M4', outDiam=outter4, coordSys=M4coord).flatMirror()

    outterD = 0.9
    Dcoord = defineCoordinate(posZ=0.237581).local()
    D = optic('D', outDiam=outterD, coordSys=Dcoord).detector()

    telescope = build.compoundOptic(M1, M2, M3, M4, D, 
                                    pupilSize=6.46, backDist=7.5,
                                    EPcoord=defineCoordinate(posZ=0).local())
    
    return telescope

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


def _outside_key(outside):
    if outside is None or (isinstance(outside, float) and np.isnan(outside)):
        return "nan"
    return float(outside)


def _as_basis_stack(basis_set, nterms):
    """Accept basis_vectors output (nterms, H, W) or a list of 2D modes."""
    basis_stack = np.asarray(basis_set)
    if basis_stack.ndim == 2:
        basis_stack = basis_stack[np.newaxis, ...]
    elif basis_stack.ndim != 3:
        basis_stack = np.stack(basis_set)
    if basis_stack.shape[0] != nterms:
        raise ValueError(
            f"Expected {nterms} basis modes, got shape {basis_stack.shape}"
        )
    return basis_stack


@lru_cache(maxsize=16)
def _polar_grid(npix, zero):
    x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.0) / zero
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    return rho, theta


@lru_cache(maxsize=16)
def basis_vectors_cached(nterms=15, npix=256, zero=19.053609871416903, outside="nan"):
    """Cached basis on a shared (rho, theta) grid. `outside`: 'nan' or a float."""
    outside_val = np.nan if outside == "nan" else float(outside)
    rho, theta = _polar_grid(npix, zero)
    return basis_vectors(
        nterms=nterms, rho=rho, theta=theta, outside=outside_val, zero=zero
    )


def clear_basis_cache():
    basis_vectors_cached.cache_clear()
    _polar_grid.cache_clear()


def decompose_psf(psf, nterms=15, basis=basis_vectors, **kwargs):
    npix = psf.shape[0]
    basis_stack = _as_basis_stack(
        basis(nterms=nterms, npix=npix, outside=np.nan, **kwargs), nterms
    )
    basis_flat = basis_stack.reshape(nterms, -1)
    psf_flat = psf.ravel()
    numerators = basis_flat @ psf_flat
    denominators = np.einsum("ij,ij->i", basis_flat, basis_flat)
    return numerators / denominators


def decompose_psf_cached(psf, nterms=15, outside=np.nan, **kwargs):
    npix = psf.shape[0]
    zero = kwargs.get("zero", 19.053609871416903)
    basis_stack = basis_vectors_cached(
        nterms=nterms, npix=npix, zero=zero, outside=_outside_key(outside)
    )
    basis_flat = basis_stack.reshape(nterms, -1)
    psf_flat = psf.ravel()
    numerators = basis_flat @ psf_flat
    denominators = np.einsum("ij,ij->i", basis_flat, basis_flat)
    return numerators / denominators

def genpsf(WL=650e-9, pdx=0, pdy=0, pdz=0, ptx=0, pty=0, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5):

    telescope = telescopeCreation(WL,pdx, pdy, pdz, ptx, pty, sdx, sdy, sdz, stx, sty)
    infoTelescope = details(telescope, fieldX=0, fieldY=0.17, wavelength=WL)
    PSF, _ = infoTelescope.fftPSF(npx=16, pad=2)
    r = decompose_psf(PSF, nterms=dterms, basis=basis_vectors)

    return r


# Cached wrapper around `genpsf` to avoid rebuilding identical PSFs repeatedly.
# Quantize input floats to reduce cache cardinality and improve hit rate.
@lru_cache(maxsize=1024)
def _genpsf_cached(qWL, qpdx, qpdy, qpdz, qptx, qpty, qsdx, qsdy, qsdz, qstx, qsty, qdterms):
    return genpsf(WL=qWL, pdx=qpdx, pdy=qpdy, pdz=qpdz, ptx=qptx, pty=qpty,
                  sdx=qsdx, sdy=qsdy, sdz=qsdz, stx=qstx, sty=qsty, dterms=qdterms)

def genpsf_cached(WL=650e-9, pdx=0, pdy=0, pdz=0, ptx=0, pty=0, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5, precision=3):
    q = lambda x: round(float(x), precision)
    return _genpsf_cached(q(WL), q(pdx), q(pdy), q(pdz), q(ptx), q(pty), q(sdx), q(sdy), q(sdz), q(stx), q(sty), int(dterms))

C = np.array([[ 1.16634361e-05, -2.32559604e-06, -1.38717321e-06],
              [-2.32559604e-06,  2.10292548e-06,  1.61782429e-07],
              [-1.38717321e-06,  1.61782429e-07,  1.31161567e-06]])
Cinv = np.linalg.inv(C)  # precision matrix
lnorm = -0.5 * (np.log(2 * np.pi) * 3 + np.log(np.linalg.det(C)))  # ln(normalization)

def validator(xy, iter=30):
    x = xy[0]
    y = xy[1]
    ct = genpsf_cached(650e-9, x, y, 0, 0, 0, 0, 0, 0, 0, 0, 3, precision=3)
    
    # log-likelihood
    def loglike(theta):
        pdx,pdy = theta
        ## PSF
        c = genpsf_cached(WL=650e-9, pdx=pdx, pdy=pdy, dterms=3, precision=3)

        return -0.5 * np.dot(ct-c, np.dot(Cinv, ct-c)) + lnorm

    # prior transform
    def ptform(utheta):

        upx,upy = utheta

        mpx, spx = 0, 16  # mean and standard deviation
        lpx, hpx = -50, 50  # lower and upper bounds
        xpx = scipy.stats.truncnorm.ppf(upx, lpx, hpx, loc=mpx, scale=spx)

        mpy, spy = 0, 25  # mean and standard deviation
        lpy, hpy = -100, 100  # lower and upper bounds
        xpy = scipy.stats.truncnorm.ppf(upy, lpy, hpy, loc=mpy, scale=spy)

        return xpx,xpy
    
    dsampler = dynesty.DynamicNestedSampler(loglike, ptform, ndim=2, 
                                  sample='unif')
    dsampler.run_nested(maxiter=iter)
    dres = dsampler.results

    t0 = np.array([np.median(dres.samples[:,0]), np.median(dres.samples[:,1])])

    def lstep(t):
        return np.sum((genpsf_cached(WL=650e-9, pdx=x, pdy=y, dterms=3, precision=3) - genpsf_cached(WL=650e-9, pdx=t[0], pdy=t[1], dterms=3, precision=3))**2)
    
    res = minimize(lstep, t0, method='Nelder-Mead', tol = 1e-6, options={'disp': True})

    return t0, np.round(res.x, 3), np.sum(np.round(res.x, 3)-xy)**2

def valoop(xy):
    g300, val300, e300 = validator(xy, iter=300)
    g30, val30, e30 = validator(xy)
    if np.sum(val300 - val30) != 0:
        g3000, val3000, e3000 = validator(xy, iter=3000)
        print(xy, g3000, val3000, e3000)
        with open('validate.txt', 'a') as file:
            file.write(f"{xy}, {g3000}, {val3000}, {e3000}\n")
    else:
        print(xy, g30, val30, e30)
        with open('validate.txt', 'a') as file:
            file.write(f"{xy}, {g30}, {val30}, {e30}\n")

xy = np.zeros((10,2))
for i in range(10):
    x = np.random.normal(loc=0,scale=10)
    y = np.random.normal(loc=0,scale=20)
    ars = np.array([np.round(x,2),np.round(y,2)])
    xy[i] = ars

if __name__ == "__main__":
    with Pool(2) as p:
        p.map(valoop, xy)