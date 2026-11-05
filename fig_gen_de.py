from optiland import optic, physical_apertures
import optiland.backend as be
from optiland.psf import MMDFTPSF
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1
from scipy.special import jv
from scipy.stats import truncnorm
from scipy.optimize import differential_evolution
from poppy.zernike import zernike_basis
plt.rcParams['figure.figsize'] = (20,10)
from nautilus import Sampler
import corner
import pandas as pd
from optiland.wavefront import OPD
from matplotlib.colors import LogNorm

be.set_backend("numpy")
be.set_precision("float64")

def decompose_psf(psf, nterms=10, basis=zernike_basis, **kwargs):
    npix = psf.shape[0]
    basis_set = basis(nterms=nterms, npix=npix, outside=0.0, **kwargs)
    basis_set = np.array(basis_set)
    basis_flat = basis_set.reshape(nterms, -1)
    numerators = basis_flat @ psf.ravel()
    denominators = np.einsum('ij,ij->i', basis_flat, basis_flat)
    coeffs = numerators / denominators
    return coeffs

def telescopeCreation(WL=0.65,pdx=0,pdy=0,pdz=0,ptx=0,pty=0,sdx=0,sdy=0,sdz=0,stx=0,sty=0):

    K=1000 #mm to micron
    #Pearl
    pearl = optic.Optic(name="Pearl")

    # set aperture
    pearl.set_aperture(aperture_type="EPD", value=6471.26)
    aperture1 = physical_apertures.RadialAperture(r_min=690, r_max=3210) 
    a1 = physical_apertures.PolygonAperture(x=[550, 3210, 3210, 550], y=[300, 300, -300, -300])
    a2 = physical_apertures.PolygonAperture(x=[-1835, -1375, 260, -260], y=[2705, 2955, 150, -150])
    a3 = physical_apertures.PolygonAperture(x=[-1835, -1375, 260, -260], y=[-2705, -2955, -150, 150])
    a = aperture1 - a1 - a2 - a3

    # set wavelengths
    pearl.wavelengths.add(value=WL)

    # add surfaces

    pearl.surfaces.add(
        index=0,
        radius=be.inf,
        thickness=be.inf,
    )

    # --- absolute mode from here: x, y, z (no thickness, no dx/dy) ---

    pearl.surfaces.add(
        index=1,
        radius=-16256,
        x=pdx / K,
        y=pdy / K,
        z=0 + pdz / K,              # independent M1 piston
        rx=np.deg2rad(ptx),
        ry=np.deg2rad(pty),
        conic=-0.995357,
        material="mirror",
        is_stop=True,
    )
    pearl.surfaces.add(
        index=2,
        radius=-1660.575,
        x=sdx / K,
        y=sdy / K,
        z=-7400.005 + sdz / K,              # independent M2 piston
        rx=np.deg2rad(stx),
        ry=np.deg2rad(sty),
        conic=-1.566503,
        material="mirror",
    )
    pearl.surfaces.add(
        index=3,
        radius=-1830.5174,
        x=0.0, y=0.0, z=49.965,          # M3 fixed in lab frame
        conic=-0.718069,
        material="mirror",
    )
    pearl.surfaces.add(
        index=4,
        radius=be.inf,
        x=0.0, y=0.0, z=-983.655,          # M4 fixed
        material="mirror",
    )
    pearl.surfaces.add(
        index=5,
        x=0.0, y=0.0, z=236.685,          # image / detector
    )
    pearl.fields.set_type(field_type="angle")
    pearl.fields.add(y=0.17)
    pearl.fields.add(y=0.285)
    pearl.fields.add(y=0.055)

    return pearl

def genpsf(WL=0.650, pdx=0, pdy=0, pdz=0, ptx=0, pty=0, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5, field = (0,0)):

    telescope = telescopeCreation(WL, pdx, pdy, pdz, ptx, pty, sdx, sdy, sdz, stx, sty)
    psf = MMDFTPSF(telescope, field=field, wavelength=WL, num_rays=512)
    PSF = psf.psf[384:640,384:640]
    r = decompose_psf(PSF, nterms=dterms)

    return r

def genpsfn(WL=0.650, pdx=0, pdy=0, pdz=0, ptx=0, pty=0, sdx=0, sdy=0, sdz=0, stx=0, sty=0, dterms=5, field = (0,0)):

    telescope = telescopeCreation(WL, pdx, pdy, pdz, ptx, pty, sdx, sdy, sdz, stx, sty)
    psf = MMDFTPSF(telescope, field=field, wavelength=WL, num_rays=512)
    PSF = psf.psf[384:640,384:640]
    gauss_noise = np.random.normal(0, 0.01, PSF.shape)
    r = decompose_psf(PSF + gauss_noise, nterms=dterms)

    return r

#Truth

pdx_true = -152.75016400314658#truncnorm(-800, 800,loc=0,scale=250).rvs()
pdy_true =  316.9119521175452#truncnorm(-800, 800,loc=0,scale=250).rvs()
pdz_true =  26.63305492799793#truncnorm(-800, 800,loc=0,scale=50).rvs()
sdx_true =  -42.31026973120823#truncnorm(-2500, 2500,loc=0,scale=800).rvs()
sdy_true =  923.3301076415017#truncnorm(-2500, 2500,loc=0,scale=800).rvs()
sdz_true =  -32.7503419412196#truncnorm(-2500, 2500,loc=0,scale=100).rvs()
ptx_true =  2.9825886165449316e-05#truncnorm(-0.01, 0.01,loc=0,scale=0.003).rvs()
pty_true =  1.6083989287690292e-05#truncnorm(-0.01, 0.01,loc=0,scale=0.003).rvs()
stx_true =  0.02694973236912114#truncnorm(-0.4, 0.4,loc=0,scale=0.13).rvs()
sty_true =  0.009802980828159563#truncnorm(-0.4, 0.4,loc=0,scale=0.13).rvs()

bounds = [(-711.01454326, 634.62883661),
          (-320.14994557, 760.93448734),
          (-131.74872611, 341.54114045),
          (-1510.14805135, 1569.58463458),
          (-522.39560624, 2387.03873492),
          (-187.49604503, 277.52959379),
          (-0.00894239, 0.00924757),
          (-0.00987663, 0.00883954),
          (0.00714548, 0.03869936),
          (0.00106751, 0.02105905)]

x0 = [-156.08514143, 329.45839292, 59.52159352, -52.88489196, 852.9971309, 1.25561018, -0.00026229, 0.00082091, 0.0257526, 0.00998523]

def my_callback(intermediate_result):
    # intermediate_result is an OptimizeResult object
    with open("solutions_logtma.txt", "a") as f:
        f.write(f"Objective: {intermediate_result.fun}, Best X: {intermediate_result.x}\n")
    if intermediate_result.fun <= 1e-15:
        print(f"Terminating: reached objective value {intermediate_result.fun}")
        return True  # Returns True to halt optimization
    return False

def lstep(t, ct, ctf10, ctf01, ctf11, ctfm10, ctf0m1, ctfm11, ctf1m1, ctfm1m1, ctf05):
      return np.sum(np.array(ct - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(0, 0.5965)))**2 + np.array(ctf10 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(1,0)))**2 + np.array(ctf01 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(0,1)))**2 + np.array(ctf11 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(1,1)))**2 + np.array(ctfm10 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(-1,0)))**2 + np.array(ctf0m1 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(0,-1)))**2 + np.array(ctfm11 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(-1,1)))**2 + np.array(ctf1m1 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(1,-1)))**2 + np.array(ctfm1m1 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(-1,-1)))**2 + np.array(ctf05 - genpsf(WL=0.65, pdx=t[0], pdy=t[1], pdz=t[2], sdx=t[3], sdy=t[4], sdz=t[5], ptx=t[6], pty=t[7], stx=t[8], sty=t[9], dterms=11, field=(0.5,0.5)))**2)
def main():
    ## PSF
    ct = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(0, 0.5965))
    ctf10 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(1,0))
    ctf01 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(0,1))
    ctf11 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(1,1))
    ctfm10 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(-1,0))
    ctf0m1 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(0,-1))
    ctfm11 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(-1,1))
    ctf1m1 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(1,-1))
    ctfm1m1 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(-1,-1))
    ctf05 = genpsfn(WL=0.65, pdx=pdx_true, pdy=pdy_true, pdz=pdz_true, sdx=sdx_true, sdy=sdy_true, sdz=sdz_true, ptx=ptx_true, pty=pty_true, stx=stx_true, sty=sty_true, dterms=11, field=(0.5, 0.5))

    print(f"{pdx_true},\n {pdy_true},\n {pdz_true},\n {sdx_true},\n {sdy_true}, \n {sdz_true}, \n {ptx_true},\n {pty_true},\n {stx_true},\n {sty_true}")

    result = differential_evolution(lstep, bounds, args=(ct, ctf10, ctf01, ctf11, ctfm10, ctf0m1, ctfm11, ctf1m1, ctfm1m1, ctf05), callback = my_callback, x0=x0, tol=1e-10, disp=True, workers=-1, updating='deferred', polish=True, maxiter=2000)
    print(pdx_true, pdy_true, pdz_true, sdx_true, sdy_true, sdz_true, ptx_true, pty_true, stx_true, sty_true)
    print(result.x)


if __name__ == "__main__":
    main()
