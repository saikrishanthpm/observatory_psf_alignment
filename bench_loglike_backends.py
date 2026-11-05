#!/usr/bin/env python3
"""Benchmark loglike() from tma_moredof2 with optiland numpy vs torch backends."""

from __future__ import annotations

import gc
import statistics
import time
from typing import Callable

import numpy as np
import torch
from optiland import optic, physical_apertures
import optiland.backend as be
from optiland.psf import MMDFTPSF
from poppy.zernike import zernike_basis

# Fixed test parameters (representative misalignment vector)
THETA = np.array(
    [12.5, -33.0, 18.0, 210.0, -480.0, 75.0, 0.002, -0.003, 0.04, -0.06],
    dtype=np.float64,
)
WL = 0.65
DTERMS = 11
NUM_RAYS = 512
WARMUP = 1
REPEATS = 3


def _as_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def decompose_psf(psf, nterms=10, basis=zernike_basis, **kwargs):
    psf = _as_numpy(psf)
    npix = psf.shape[0]
    basis_set = np.array(basis(nterms=nterms, npix=npix, outside=0.0, **kwargs))
    basis_flat = basis_set.reshape(nterms, -1)
    denominators = np.einsum("ij,ij->i", basis_flat, basis_flat)
    numerators = basis_flat @ psf.ravel()
    return numerators / denominators


def telescopeCreation(
    WL=0.65,
    pdx=0,
    pdy=0,
    pdz=0,
    ptx=0,
    pty=0,
    sdx=0,
    sdy=0,
    sdz=0,
    stx=0,
    sty=0,
):
    K = 1000
    T = 3600
    pearl = optic.Optic(name="Pearl")
    pearl.set_aperture(aperture_type="EPD", value=6471.26)
    aperture1 = physical_apertures.RadialAperture(r_min=690, r_max=3210)
    a1 = physical_apertures.PolygonAperture(
        x=[550, 3210, 3210, 550], y=[300, 300, -300, -300]
    )
    a2 = physical_apertures.PolygonAperture(
        x=[-1835, -1375, 260, -260], y=[2705, 2955, 150, -150]
    )
    a3 = physical_apertures.PolygonAperture(
        x=[-1835, -1375, 260, -260], y=[-2705, -2955, -150, 150]
    )
    a = aperture1 - a1 - a2 - a3
    pearl.wavelengths.add(value=WL)
    pearl.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    pearl.surfaces.add(
        index=1,
        radius=-16256,
        thickness=-7400.005 + pdz / K,
        dx=pdx / K,
        dy=pdy / K,
        rx=ptx / T,
        ry=pty / T,
        conic=-0.9953569,
        material="mirror",
        aperture=a,
        is_stop=True,
    )
    pearl.surfaces.add(
        index=2,
        radius=-1660.575,
        thickness=7449.97 + sdz / K,
        dx=sdx / K,
        dy=sdy / K,
        rx=stx / T,
        ry=sty / T,
        conic=-1.566503,
        material="mirror",
    )
    pearl.surfaces.add(
        index=3,
        radius=-1830.517,
        thickness=-1033.62,
        conic=-0.718069,
        material="mirror",
    )
    pearl.surfaces.add(index=4, radius=be.inf, thickness=1220.34, material="mirror")
    pearl.surfaces.add(index=5)
    pearl.fields.set_type(field_type="angle")
    pearl.fields.add(y=0.0)
    pearl.fields.add(y=0.115)
    pearl.fields.add(y=-0.115)
    return pearl


def genpsf(
    WL=0.650,
    pdx=0,
    pdy=0,
    pdz=0,
    ptx=0,
    pty=0,
    sdx=0,
    sdy=0,
    sdz=0,
    stx=0,
    sty=0,
    dterms=5,
    field=(0, 0),
):
    telescope = telescopeCreation(WL, pdx, pdy, pdz, ptx, pty, sdx, sdy, sdz, stx, sty)
    psf = MMDFTPSF(telescope, field=field, wavelength=WL, num_rays=NUM_RAYS)
    PSF = _as_numpy(psf.psf)[384:640, 384:640]
    return decompose_psf(PSF, nterms=dterms, basis=zernike_basis)


def make_loglike(fixtures: dict) -> Callable:
    ct = fixtures["ct"]
    ctf10 = fixtures["ctf10"]
    ctf01 = fixtures["ctf01"]
    ctf11 = fixtures["ctf11"]
    ctfm10 = fixtures["ctfm10"]
    ctf0m1 = fixtures["ctf0m1"]
    ctfm11 = fixtures["ctfm11"]
    ctf1m1 = fixtures["ctf1m1"]
    ctd1 = fixtures["ctd1"]
    ctd2 = fixtures["ctd2"]
    Cinv = fixtures["Cinv"]
    Cf10inv = fixtures["Cf10inv"]
    Cf01inv = fixtures["Cf01inv"]
    Cf11inv = fixtures["Cf11inv"]
    Cfm10inv = fixtures["Cfm10inv"]
    Cf0m1inv = fixtures["Cf0m1inv"]
    Cfm11inv = fixtures["Cfm11inv"]
    Cf1m1inv = fixtures["Cf1m1inv"]
    lnorm = fixtures["lnorm"]
    lnormf10 = fixtures["lnormf10"]
    lnormf01 = fixtures["lnormf01"]
    lnormf11 = fixtures["lnormf11"]
    lnormfm10 = fixtures["lnormfm10"]
    lnormf0m1 = fixtures["lnormf0m1"]
    lnormfm11 = fixtures["lnormfm11"]
    lnormf1m1 = fixtures["lnormf1m1"]

    def loglike(theta):
        lpdx, lpdy, lpdz, lsdx, lsdy, lsdz, lptx, lpty, lstx, lsty = theta
        kw = dict(
            WL=WL,
            pdx=lpdx,
            pdy=lpdy,
            pdz=lpdz,
            sdx=lsdx,
            sdy=lsdy,
            sdz=lsdz,
            ptx=lptx,
            pty=lpty,
            stx=lstx,
            sty=lsty,
            dterms=DTERMS,
        )
        c = _as_numpy(genpsf(**kw, field=(0, 0)))
        cf10 = _as_numpy(genpsf(**kw, field=(1, 0)))
        cf01 = _as_numpy(genpsf(**kw, field=(0, 1)))
        cf11 = _as_numpy(genpsf(**kw, field=(1, 1)))
        cfm10 = _as_numpy(genpsf(**kw, field=(-1, 0)))
        cf0m1 = _as_numpy(genpsf(**kw, field=(0, -1)))
        cfm11 = _as_numpy(genpsf(**kw, field=(-1, 1)))
        cf1m1 = _as_numpy(genpsf(**kw, field=(1, -1)))
        kw_d1 = {**kw, "pdz": lpdz + 50}
        kw_d2 = {**kw, "sdz": lsdz + 500}
        cd1 = _as_numpy(genpsf(**kw_d1, field=(0, 0)))
        cd2 = _as_numpy(genpsf(**kw_d2, field=(0, 0)))

        return (
            -0.5 * np.matmul(c - ct, np.matmul(Cinv, c - ct))
            + lnorm
            - 0.5 * np.matmul(cf10 - ctf10, np.matmul(Cf10inv, cf10 - ctf10))
            + lnormf10
            - 0.5 * np.matmul(cf01 - ctf01, np.matmul(Cf01inv, cf01 - ctf01))
            + lnormf01
            - 0.5 * np.matmul(cf11 - ctf11, np.matmul(Cf11inv, cf11 - ctf11))
            + lnormf11
            - 0.5 * np.matmul(cfm10 - ctfm10, np.matmul(Cfm10inv, cfm10 - ctfm10))
            + lnormfm10
            - 0.5 * np.matmul(cf0m1 - ctf0m1, np.matmul(Cf0m1inv, cf0m1 - ctf0m1))
            + lnormf0m1
            - 0.5 * np.matmul(cfm11 - ctfm11, np.matmul(Cfm11inv, cfm11 - ctfm11))
            + lnormfm11
            - 0.5 * np.matmul(cf1m1 - ctf1m1, np.matmul(Cf1m1inv, cf1m1 - ctf1m1))
            + lnormf1m1
            - 0.5 * np.matmul(cd1 - ctd1, np.matmul(Cinv, cd1 - ctd1))
            + lnorm
            - 0.5 * np.matmul(cd2 - ctd2, np.matmul(Cinv, cd2 - ctd2))
            + lnorm
        )

    return loglike


def _load_covariance_fixtures() -> dict:
    """Load precomputed C matrices from notebook cell (same as tma_moredof2)."""
    # Minimal diagonal covariances for benchmark if import fails — algebra is negligible.
    # Prefer notebook values when available via running under saved npz.
    npz_path = __file__.replace("bench_loglike_backends.py", "bench_loglike_fixtures.npz")
    try:
        data = np.load(npz_path)
        return {k: data[k] for k in data.files}
    except FileNotFoundError:
        pass

    rng = np.random.default_rng(0)
    def cov():
        A = rng.normal(size=(DTERMS, DTERMS)) * 1e-3
        return A @ A.T + np.eye(DTERMS) * 1e-4

    mats = [cov() for _ in range(8)]
    C = mats[0]
    invs = [np.linalg.inv(m) for m in mats]
    lns = [-0.5 * (np.log(2 * np.pi) * DTERMS + np.log(np.linalg.det(m))) for m in mats]
    z = np.zeros(DTERMS)
    return {
        "ct": z,
        "ctf10": z,
        "ctf01": z,
        "ctf11": z,
        "ctfm10": z,
        "ctf0m1": z,
        "ctfm11": z,
        "ctf1m1": z,
        "ctd1": z,
        "ctd2": z,
        "Cinv": invs[0],
        "Cf10inv": invs[1],
        "Cf01inv": invs[2],
        "Cf11inv": invs[3],
        "Cfm10inv": invs[4],
        "Cf0m1inv": invs[5],
        "Cfm11inv": invs[6],
        "Cf1m1inv": invs[7],
        "lnorm": lns[0],
        "lnormf10": lns[1],
        "lnormf01": lns[2],
        "lnormf11": lns[3],
        "lnormfm10": lns[4],
        "lnormf0m1": lns[5],
        "lnormfm11": lns[6],
        "lnormf1m1": lns[7],
    }


def configure_backend(name: str, precision: str):
    be.set_backend(name)
    be.set_precision(precision)
    if name == "torch":
        torch.set_default_device("cpu")


def bench_backend(name: str, precision: str, fixtures: dict) -> dict:
    configure_backend(name, precision)
    loglike = make_loglike(fixtures)
    gc.collect()

    for _ in range(WARMUP):
        loglike(THETA)

    times = []
    values = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        values.append(float(loglike(THETA)))
        times.append(time.perf_counter() - t0)

    return {
        "backend": name,
        "precision": precision,
        "times_s": times,
        "mean_s": statistics.mean(times),
        "stdev_s": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "loglike": values[-1],
    }


def main():
    print(f"Benchmark: loglike (tma_moredof2), num_rays={NUM_RAYS}, repeats={REPEATS}")
    print(f"theta = {THETA}\n")

    fixtures = _load_covariance_fixtures()

    configs = [
        ("numpy", "float64"),
        ("torch", "float32"),
        ("torch", "float64"),
    ]

    results = []
    for backend, precision in configs:
        print(f"Running {backend} / {precision} ...", flush=True)
        try:
            r = bench_backend(backend, precision, fixtures)
            results.append(r)
            print(
                f"  mean={r['mean_s']:.3f}s  stdev={r['stdev_s']:.3f}s  "
                f"loglike={r['loglike']:.4f}\n"
            )
        except Exception as e:
            print(f"  FAILED: {e}\n")
            results.append(
                {"backend": backend, "precision": precision, "error": str(e)}
            )

    ok = [r for r in results if "mean_s" in r]
    if len(ok) >= 2:
        fastest = min(ok, key=lambda r: r["mean_s"])
        print("=" * 60)
        print(f"Fastest: {fastest['backend']} ({fastest['precision']}) — {fastest['mean_s']:.3f}s per loglike")
        for r in ok:
            ratio = r["mean_s"] / fastest["mean_s"]
            print(
                f"  {r['backend']:5s} {r['precision']:8s}: "
                f"{r['mean_s']:.3f}s  ({ratio:.2f}x)"
            )


if __name__ == "__main__":
    main()
