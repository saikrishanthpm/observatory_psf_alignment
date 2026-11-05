from optiland import optic, physical_apertures
import optiland.backend as be
from optiland.psf import MMDFTPSF
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution
import torch
plt.rcParams['figure.figsize'] = (20,10)
import warnings
warnings.filterwarnings('ignore')
from nautilus import Sampler

from poppy.zernike import zernike_basis
from functools import lru_cache

be.set_backend("torch")
be.grad_mode.enable()
be.set_precision("float64")

torch.set_default_dtype(torch.float64)

def decompose_psf(psf, nterms=10, basis=zernike_basis, **kwargs):
    npix = psf.shape[0]
    basis_set = basis(nterms=nterms, npix=npix, outside=0.0, **kwargs)
    basis_set = torch.tensor(basis_set)
    psf_flat = psf.ravel()
    basis_flat = basis_set.reshape(nterms, -1).to(torch.float64)
    numerators = torch.matmul(basis_flat, psf_flat)
    denominators = torch.einsum('ij,ij->i', basis_flat, basis_flat)
    coeffs = numerators / denominators
    return coeffs.detach().clone()

def hubble(WL=0.55,sdx=0,sdy=0,sdz=0,stx=0,sty=0):

    K=1000 #mm to micron
    T = 3600
    #Hubble
    hubble = optic.Optic(name="Hubble")

    hubble.add_surface(index=0, radius=np.inf, thickness=np.inf)
    hubble.add_surface(index=1, thickness=4910.01016)

    # add physical aperture in primary mirror
    obscuration = physical_apertures.RadialAperture(r_max=1200, r_min=384)
    c1 = physical_apertures.OffsetRadialAperture(r_max=75, r_min=0, offset_x=0, offset_y=1000)
    c2 = physical_apertures.OffsetRadialAperture(r_max=75, r_min=0, offset_x=-707.15, offset_y=-707.15)
    c3 = physical_apertures.OffsetRadialAperture(r_max=75, r_min=0, offset_x=707.15, offset_y=-707.15)
    r1 = physical_apertures.RectangularAperture(x_min=-19, x_max=19, y_min=-1200, y_max=1200)
    r2 = physical_apertures.RectangularAperture(x_min=-1200, x_max=1200, y_min=-19, y_max=19)

    ap = obscuration - r1 - r2 -c1 - c2 - c3

    hubble.add_surface(
            index=2,
            radius=-11040.02286,
            thickness=-4910.01016,
            material="mirror",
            is_stop=True,
            conic=-1.001152,
            aperture=ap,
        )
    hubble.add_surface(
            index=3,
            radius=-1349.31166,
            thickness=6365.20955+sdz/K,
            dx=sdx/K,
            dy=sdy/K,
            rx=stx/T,
            ry=sty/T,
            material="mirror",
            conic=-1.483014,
        )
    hubble.add_surface(index=4, radius=-635.38227)

    hubble.set_aperture(aperture_type="EPD", value=2400)

    hubble.set_field_type(field_type="angle")
    hubble.add_field(y=0.)
    hubble.add_field(y=0.038)
    hubble.add_field(y=-0.038)

    hubble.add_wavelength(value=WL, is_primary=True)
    
    return hubble

def genpsf(WL=0.55, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5):

    telescope = hubble(WL, sdx, sdy, sdz, stx, sty)
    psf = MMDFTPSF(telescope, field=(0,0), wavelength=WL, grid_size=64)
    PSF = psf.psf[96:160,96:160]
    r = decompose_psf(PSF, nterms=dterms, basis=zernike_basis)

    return r


# Cache wrapper for genpsf to avoid recomputing identical PSFs.
@lru_cache(maxsize=1024)
def _genpsf_cached(qWL, qsdx, qsdy, qsdz, qstx, qsty, qdterms):
  return genpsf(WL=qWL, sdx=qsdx, sdy=qsdy, sdz=qsdz, stx=qstx, sty=qsty, dterms=qdterms)

def genpsf_cached(WL=0.55, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5, precision=3):
  q = lambda x: round(float(x), precision)
  return _genpsf_cached(q(WL), q(sdx), q(sdy), q(sdz), q(stx), q(sty), int(dterms))

C = torch.tensor([[ 2.8546e-02,  2.4201e-04, -3.1598e-04, -4.8934e-02, -3.2693e-04,
         -5.5986e-04, -1.0243e-03, -1.7370e-03, -4.9009e-05, -3.2011e-05,
          3.9497e-02,  9.0271e-04,  4.9170e-04,  1.0310e-04,  6.6334e-05],
        [ 2.4201e-04,  9.9158e-02, -1.7614e-03,  1.1474e-03,  2.0232e-03,
          2.0911e-03,  1.9443e-03, -9.4698e-02, -6.1838e-04,  1.3896e-05,
         -2.2020e-03, -2.8686e-03, -2.5514e-03,  1.9861e-05, -1.4131e-04],
        [-3.1598e-04, -1.7614e-03,  1.0099e-01,  2.1634e-03,  2.4404e-03,
         -2.2755e-03, -9.7984e-02,  1.9599e-03,  3.0645e-04,  2.1593e-04,
         -3.0290e-03,  2.9498e-03, -3.1703e-03, -7.1629e-05, -5.2912e-05],
        [-4.8934e-02,  1.1474e-03,  2.1634e-03,  1.0426e-01,  3.6707e-04,
          7.3733e-04,  8.9616e-04,  2.2155e-03,  3.6045e-05,  2.3623e-06,
         -9.9253e-02, -1.3523e-03, -5.8528e-04, -2.5646e-04, -9.3452e-05],
        [-3.2693e-04,  2.0232e-03,  2.4404e-03,  3.6707e-04,  3.4768e-02,
         -2.5238e-04, -1.7755e-03, -1.2087e-03,  1.0214e-03, -1.1731e-03,
         -4.0595e-05,  1.7350e-04, -3.5174e-02, -3.1900e-05,  1.4332e-04],
        [-5.5986e-04,  2.0911e-03, -2.2755e-03,  7.3733e-04, -2.5238e-04,
          3.5909e-02,  1.5078e-03, -1.4558e-03,  9.2375e-04,  1.1446e-03,
         -3.4590e-04, -3.7191e-02,  1.6025e-04,  2.9694e-04, -1.3420e-04],
        [-1.0243e-03,  1.9443e-03, -9.7984e-02,  8.9616e-04, -1.7755e-03,
          1.5078e-03,  1.1243e-01, -2.1993e-03, -1.7086e-04, -3.8854e-04,
          5.2096e-04, -2.7584e-03,  3.1177e-03,  7.1545e-05,  5.9634e-05],
        [-1.7370e-03, -9.4698e-02,  1.9599e-03,  2.2155e-03, -1.2087e-03,
         -1.4558e-03, -2.1993e-03,  1.0681e-01,  6.0168e-04, -8.2099e-05,
         -6.8326e-04,  2.7540e-03,  2.2908e-03, -3.5036e-05,  1.4671e-04],
        [-4.9009e-05, -6.1838e-04,  3.0645e-04,  3.6045e-05,  1.0214e-03,
          9.2375e-04, -1.7086e-04,  6.0168e-04,  9.0955e-03, -1.2108e-04,
          1.7585e-05, -7.1916e-04, -8.8275e-04, -3.6350e-04,  3.9637e-04],
        [-3.2011e-05,  1.3896e-05,  2.1593e-04,  2.3623e-06, -1.1731e-03,
          1.1446e-03, -3.8854e-04, -8.2099e-05, -1.2108e-04,  9.4166e-03,
          5.8236e-06, -1.0284e-03,  1.0278e-03,  3.9824e-04,  3.7200e-04],
        [ 3.9497e-02, -2.2020e-03, -3.0290e-03, -9.9253e-02, -4.0595e-05,
         -3.4590e-04,  5.2096e-04, -6.8326e-04,  1.7585e-05,  5.8236e-06,
          1.0623e-01,  6.7788e-04,  1.9345e-04,  2.4776e-04,  5.0283e-05],
        [ 9.0271e-04, -2.8686e-03,  2.9498e-03, -1.3523e-03,  1.7350e-04,
         -3.7191e-02, -2.7584e-03,  2.7540e-03, -7.1916e-04, -1.0284e-03,
          6.7788e-04,  4.3718e-02, -1.6952e-04, -2.9792e-04,  1.8890e-04],
        [ 4.9170e-04, -2.5514e-03, -3.1703e-03, -5.8528e-04, -3.5174e-02,
          1.6025e-04,  3.1177e-03,  2.2908e-03, -8.8275e-04,  1.0278e-03,
          1.9345e-04, -1.6952e-04,  4.0530e-02, -2.1973e-06, -1.8618e-04],
        [ 1.0310e-04,  1.9861e-05, -7.1629e-05, -2.5646e-04, -3.1900e-05,
          2.9694e-04,  7.1545e-05, -3.5036e-05, -3.6350e-04,  3.9824e-04,
          2.4776e-04, -2.9792e-04, -2.1973e-06,  1.0394e-03,  1.3562e-05],
        [ 6.6334e-05, -1.4131e-04, -5.2912e-05, -9.3452e-05,  1.4332e-04,
         -1.3420e-04,  5.9634e-05,  1.4671e-04,  3.9637e-04,  3.7200e-04,
          5.0283e-05,  1.8890e-04, -1.8618e-04,  1.3562e-05,  9.7441e-04]])
Cinv = torch.linalg.inv(C).to(torch.float64)  # precision matrix
lnorm = -0.5 * (torch.log(torch.tensor(2*torch.pi)) * 4 + torch.log(torch.linalg.det(C)))  # ln(normalization)

# log-likelihood
def loglike(theta):
    lsdx,lsdy,lstx,lsty = theta
    ## PSF
  c = genpsf_cached(WL=0.55, sdx=lsdx, sdy=lsdy, stx=lstx, sty=lsty, dterms=15, precision=3)

    return -0.5 * torch.matmul(torch.tensor(c-ct), torch.matmul(Cinv, torch.tensor(c-ct))) + lnorm

# prior transform
def ptform(utheta):

    usx,usy,ustx,usty = utheta

    xsx =  np.round((500. * (2. * usx - 1.)), decimals=2)
    xsy =  np.round((500. * (2. * usy - 1.)), decimals=2)
    xstx =  np.round((10. * (2. * ustx - 1.)), decimals=2)
    xsty =  np.round((10. * (2. * usty - 1.)), decimals=2)

    return xsx,xsy,xstx,xsty

#Truth

sdx_true = np.round(np.random.normal(loc=0,scale=50), 2)
sdy_true = np.round(np.random.normal(loc=0,scale=50), 2)
stx_true = np.round(np.random.normal(loc=0,scale=3), 2)
sty_true = np.round(np.random.normal(loc=0,scale=3), 2)

## PSF 
ct = genpsf_cached(WL=0.55, sdx=sdx_true, sdy=sdy_true, stx=stx_true, sty=sty_true, dterms=15, precision=3)

it = 0
n_it  = 2

while it<n_it:
    
    sampler = Sampler(ptform, loglike, n_dim=4)
    sampler.run(verbose=True)
    points, log_w, log_l = sampler.posterior()

    wts = np.exp(log_w)

    print(sdx_true, sdy_true, stx_true, sty_true)

    print(np.quantile(points[-100:,0], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf'),
          np.quantile(points[-100:,1], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf'),
          np.quantile(points[-100:,2], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf'),
          np.quantile(points[-100:,3], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf'))

    sdx_true -= np.quantile(points[-100:,0], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf')
    sdy_true -= np.quantile(points[-100:,1], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf')
    stx_true -= np.quantile(points[-100:,2], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf')
    sty_true -= np.quantile(points[-100:,3], 0.5, weights=np.exp(log_w)[-100:], method='inverted_cdf')
    ## PSF 
    ct = genpsf(WL=0.55, sdx=sdx_true, sdy=sdy_true, stx=stx_true, sty=sty_true, dterms=15)

    it += 1

def lstep(t):
    return np.sum(np.array(ct - genpsf(WL=0.55, sdx=t[0], sdy=t[1], stx=t[2], sty=t[3], dterms=15))**2)

bounds = [(np.quantile(points[-100:,0], 0.16, weights=np.exp(log_w)[-100:], method='inverted_cdf'),np.quantile(points[-100:,0], 0.84, weights=np.exp(log_w)[-100:], method='inverted_cdf')), (np.quantile(points[-100:,1], 0.16, weights=np.exp(log_w)[-100:], method='inverted_cdf'),np.quantile(points[-100:,1], 0.84, weights=np.exp(log_w)[-100:], method='inverted_cdf')), (np.quantile(points[-100:,2], 0.16, weights=np.exp(log_w)[-100:], method='inverted_cdf'), np.quantile(points[-100:,2], 0.84, weights=np.exp(log_w)[-100:], method='inverted_cdf')), (np.quantile(points[-100:,3], 0.16, weights=np.exp(log_w)[-100:], method='inverted_cdf'), np.quantile(points[-100:,3], 0.84, weights=np.exp(log_w)[-100:], method='inverted_cdf'))]
          
result = differential_evolution(lstep, bounds, disp=True, maxiter=1000)
print(sdx_true, sdy_true, stx_true, sty_true)
print(result.x)