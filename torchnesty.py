from optiland import optic, physical_apertures
import optiland.backend as be
from optiland import analysis
from optiland import wavefront
from optiland.psf import MMDFTPSF, FFTPSF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from torch.special import bessel_j1
from torch.special import bessel_j0
from scipy.special import j1
from scipy.special import jv
from dynesty import plotting as dyplot
from gluonts.torch.distributions import TruncatedNormal as truncnorm
import torch
from multiprocess.pool import Pool
from scipy.optimize import differential_evolution
from poppy.zernike import zernike_basis
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "1"

plt.rcParams['figure.figsize'] = (20,10)
from nautilus import Sampler
import corner

be.set_backend("torch")
be.grad_mode.enable()
be.set_precision("float64")

torch.set_default_dtype(torch.float64)

import warnings
warnings.filterwarnings('ignore')

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

def telescopeCreation(WL=0.65,pdx=0,pdy=0,pdz=0,ptx=0,pty=0,sdx=0,sdy=0,sdz=0,stx=0,sty=0):

    K=1000 #mm to micron
    T = 3600
    #Pearl
    pearl = optic.Optic(name="Pearl")

    # set aperture
    pearl.set_aperture(aperture_type="EPD", value=6471.26)
    aperture1 = physical_apertures.RadialAperture(r_min=1380, r_max=6420) 
    a1 = physical_apertures.PolygonAperture(x=[1380, 6420, 6420, 1380], y=[300, 300, -300, -300])
    a2 = physical_apertures.PolygonAperture(x=[-3469, -2949, 260, -260], y=[5410, 5710, 150, -150])
    a3 = physical_apertures.PolygonAperture(x=[-3469, -2949, 260, -260], y=[-5410, -5710, -150, 150])
    a = aperture1 - a1 - a2 - a3

    # set fields
    pearl.set_field_type(field_type="angle")
    pearl.add_field(y=0.17)

    # set wavelengths
    pearl.add_wavelength(value=WL)

    # add surfaces
    pearl.add_surface(index=0, radius=be.inf, thickness=be.inf)

    pearl.add_surface(
        index=1,
        radius=-16256,
        thickness=-7400.005+pdz/K,
        dx=pdx/K,
        dy=pdy/K,
        rx=ptx/T,
        ry=pty/T,
        conic=-0.9953569,
        material="mirror",
        aperture=a,
        is_stop=True
    )
    pearl.add_surface(
        index=2,
        radius=-1660.575,
        thickness=7449.97+sdz/K,
        dx=sdx/K,
        dy=sdy/K,
        rx=stx/T,
        ry=sty/T,
        conic=-1.566503,
        material="mirror",
        #aperture=aperture2
    )
    pearl.add_surface(
        index=3,
        radius=-1830.517,
        thickness=-1033.62,
        conic=-0.718069,
        material="mirror",

    )

    pearl.add_surface(
        index=4,
        radius=be.inf,
        thickness=1220.34,
        material="mirror",

    )

    pearl.add_surface(index=5)
    pearl.set_field_type(field_type="angle")
    pearl.add_field(y=0.)
    pearl.add_field(y=0.115)
    pearl.add_field(y=-0.115)

    return pearl

def genpsf(WL=0.650, pdx=0, pdy=0, pdz=0, ptx=0, pty=0, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5, field = (0,0)):

    telescope = telescopeCreation(WL, pdx, pdy, pdz, ptx, pty, sdx, sdy, sdz, stx, sty)
    psf = MMDFTPSF(telescope, field=field, wavelength=WL, grid_size=64)
    PSF = psf.psf[96:160,96:160]
    r = decompose_psf(PSF, nterms=dterms, basis=zernike_basis)

    return r

#Truth

pdx_true = torch.normal(mean=torch.tensor(0, dtype=torch.float64),std=torch.tensor(50, dtype=torch.float64))
pdy_true = torch.normal(mean=torch.tensor(0, dtype=torch.float64),std=torch.tensor(50, dtype=torch.float64))
sdx_true = torch.normal(mean=torch.tensor(0, dtype=torch.float64),std=torch.tensor(50, dtype=torch.float64))
sdy_true = torch.normal(mean=torch.tensor(0, dtype=torch.float64),std=torch.tensor(50, dtype=torch.float64))

## PSF
ct = genpsf(WL=0.65, pdx=pdx_true, pdy=pdy_true, sdx=sdx_true, sdy=sdy_true, dterms=11)
ctf10 = genpsf(WL=0.65, pdx=pdx_true, pdy=pdy_true, sdx=sdx_true, sdy=sdy_true, dterms=11, field=(1,0))
ctf01 = genpsf(WL=0.65, pdx=pdx_true, pdy=pdy_true, sdx=sdx_true, sdy=sdy_true, dterms=11, field=(0,1))
ctf11 = genpsf(WL=0.65, pdx=pdx_true, pdy=pdy_true, sdx=sdx_true, sdy=sdy_true, dterms=11, field=(1,1))

bounds = bounds = [(-800, 800),(-800, 800),(-2500, 2500),(-2500, 2500)]

def lstep(t):
      return np.sum(np.array(ct - genpsf(WL=0.65, pdx=t[0], pdy=t[1], sdx=t[2], sdy=t[3], dterms=11))**2 + np.array(ctf10 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], sdx=t[2], sdy=t[3], dterms=11, field=(1,0)))**2 + np.array(ctf01 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], sdx=t[2], sdy=t[3], dterms=11, field=(0,1)))**2 + np.array(ctf11 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], sdx=t[2], sdy=t[3], dterms=11, field=(1,1)))**2)

def my_callback(intermediate_result):
    # intermediate_result is an OptimizeResult object
    with open("solutions_log.txt", "a") as f:
        f.write(f"Objective: {intermediate_result.fun}, Best X: {intermediate_result.x}\n")
    #best_x = intermediate_result.x
    #current_min = intermediate_result.fun
    #print(f"Current best: {current_min} at {best_x}")

if __name__ == "__main__":
  result = differential_evolution(lstep, bounds, workers=-1, callback = my_callback, maxiter=1000)
  print(pdx_true, pdy_true, sdx_true, sdy_true)
  print(result.x)