"""
noise_analysis.py -- how the PSF noise propagates into the basis coefficients,
the likelihood and the DE point estimate, for both telescopes (thesis Chapter 5).

Both studies add zero-mean Gaussian noise (sigma = 0.01 per pixel, in Optiland
PSF units in which the aligned PSF peaks at 100) to the cropped observed PSFs
only. For each model, at the injected truth, this script computes:

* N, the coefficient covariance of the pixel noise. The coefficients are
  projections c_k = <B_k, I> / ||B_k||^2, so
  N_jk = s^2 <B_j, B_k> / (||B_j||^2 ||B_k||^2), the same at every field.
* sum_f tr(Sigma_f^-1 N): the expected value of -2 ln L at the truth minus its
  maximum, i.e. how far the noise moves the likelihood, Sigma_f being the
  calibration covariances the notebooks use.
* J, the Jacobian of the stacked coefficients (all fields) with respect to
  theta, by central differences (checked against a doubled step).
* The linearised covariance of the unweighted least-squares estimate that DE
  computes, Cov = A N A^T with A = (J^T J)^-1 J^T, and the expected minimum DE
  cost tr[(I - J A) N].
* The on-axis RMS WFE implied by that scatter, by Monte Carlo over Cov.
* A Monte Carlo sweep over noise levels: noisy observations at the truth,
  least-squares solutions found from the truth with the Jacobian held at its
  truth value, and the RMS parameter error and median WFE of those solutions.

    python noise_analysis.py            # writes noise_analysis.npz, prints a summary
"""
import json
import warnings

import numpy as np
import scipy.io as sio
from optiland.wavefront import OPD
from poppy.zernike import zernike_basis

warnings.filterwarnings("ignore")

SIGMA = 0.01
SWEEP = [0.01, 0.03, 0.1, 0.3]
N_WFE = 300            # Monte Carlo samples for the linearised WFE
N_REAL = {"hubble": 50, "tma": 30}
OUT = "noise_analysis.npz"


# ------------------------------------------------------------------ models
def _cells_defining(nb_path, starts):
    cells = json.load(open(nb_path))["cells"]
    out = []
    for key in starts:
        out.append(next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                        and "".join(c["source"]).lstrip().startswith(key)))
    return out


def hubble_model():
    """Hubble pieces taken verbatim from hubbletilus_noise.ipynb."""
    ns = {}
    for src in _cells_defining("hubbletilus_noise.ipynb", [
            "from optiland import optic", "def _is_odd", "from functools import lru_cache",
            "def hubble(", "def genpsf(", "C = np.array"]):
        exec(src, ns)
    names = ["sdx", "sdy", "sdz", "stx", "sty"]
    fields = [(0, 0), (0, 1), (1, 0), (1, 1), (-1, -1)]
    sigmas = [ns["C"], ns["Cf01"], ns["Cf10"], ns["Cf11"], ns["Cfm11"]]   # as paired in loglike
    basis = ns["basis_vectors_cached"](15, 64, 19.053609871416903, "nan")
    return dict(
        key="hubble", names=names, units=["um"] * 3 + ["deg"] * 2,
        truth=np.array([52.22, -113.86, 38.99, 0.13, -0.04]),
        fields=fields, sigmas=sigmas, basis=basis, wavelength=0.55,
        coeffs=lambda th, f: ns["genpsf"](0.55, *th, dterms=15, field=f),
        optic=lambda th: ns["hubble"](0.55, *th),
        steps=np.array([0.2, 0.2, 0.2, 2e-5, 2e-5]),
        figdata="hubble_noise_figdata.mat",
    )


def tma_model():
    """TMA pieces from fig_gen_de.py (the DE run) and tma_moredof2.ipynb (Sigma_f)."""
    import fig_gen_de as g
    ns = {"np": np}
    exec(_cells_defining("tma_moredof2.ipynb", ["C = np.array"])[0], ns)
    names = ["pdx", "pdy", "pdz", "sdx", "sdy", "sdz", "ptx", "pty", "stx", "sty"]
    fields = [(0, 0.5965), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, 1), (1, -1), (-1, -1), (0.5, 0.5)]
    sigmas = [ns[k] for k in ["C", "Cf10", "Cf01", "Cf11", "Cfm10", "Cf0m1", "Cfm11", "Cf1m1", "Cfm1m1", "Cf05"]]
    basis = np.array(zernike_basis(nterms=11, npix=256, outside=0.0))
    return dict(
        key="tma", names=names, units=["um"] * 6 + ["deg"] * 4,
        truth=np.array([getattr(g, f"{n}_true") for n in names]),
        fields=fields, sigmas=sigmas, basis=basis, wavelength=0.65,
        coeffs=lambda th, f: g.genpsf(0.65, dterms=11, field=f, **dict(zip(names, th))),
        optic=lambda th: g.telescopeCreation(0.65, **dict(zip(names, th))),
        steps=np.array([0.5] * 6 + [5e-6, 5e-6, 5e-5, 5e-5]),
        figdata="tma_fig_gen_de_figdata.mat",
    )


# ------------------------------------------------------------------ analysis
def stacked(m, th):
    return np.concatenate([np.asarray(m["coeffs"](th, f), float) for f in m["fields"]])


def jacobian(m, scale=1.0):
    th0, h = m["truth"], m["steps"] * scale
    cols = []
    for i in range(len(th0)):
        e = np.zeros_like(th0)
        e[i] = h[i]
        cols.append((stacked(m, th0 + e) - stacked(m, th0 - e)) / (2 * h[i]))
    return np.array(cols).T


def noise_cov(basis, s):
    B = basis.reshape(basis.shape[0], -1)
    G = B @ B.T
    n = np.diag(G)
    return s ** 2 * G / np.outer(n, n)


def ls_operator(J, steps):
    """A = (J^T J)^-1 J^T, computed in step-scaled units for conditioning."""
    return steps[:, None] * np.linalg.pinv(J * steps[None, :])


def wfe_fn(m):
    z_nom = OPD(m["optic"](np.zeros_like(m["truth"])), field=(0, 0),
                wavelength=m["wavelength"]).generate_opd_map()["z"]

    def wfe(delta):          # delta = theta_true - estimate
        z = OPD(m["optic"](delta), field=(0, 0), wavelength=m["wavelength"]).generate_opd_map()["z"]
        return float(np.nanstd(z_nom - z))
    return wfe


def analyse(m, rng):
    P, F = len(m["truth"]), len(m["fields"])
    J = jacobian(m)
    J2 = jacobian(m, scale=2.0)
    lin = np.abs(J2 - J).max(axis=0) / np.abs(J).max(axis=0)

    N1 = noise_cov(m["basis"], SIGMA)
    N = np.kron(np.eye(F), N1)
    A = ls_operator(J, m["steps"])
    cov = A @ N @ A.T
    sig_theta = np.sqrt(np.diag(cov))
    floor = float(np.trace((np.eye(J.shape[0]) - J @ A) @ N))
    chi2_noise = float(sum(np.trace(np.linalg.solve(S, N1)) for S in m["sigmas"]))
    ratio = np.array([np.diag(N1) / np.diag(S) for S in m["sigmas"]])

    # prior-free conditioning: singular values of J in step-scaled units
    sv = np.linalg.svd(J * m["steps"][None, :], compute_uv=False)

    wfe = wfe_fn(m)
    w, V = np.linalg.eigh(cov)
    L = V * np.sqrt(np.clip(w, 0, None))
    wfe_lin = np.array([wfe(L @ rng.standard_normal(P)) for _ in range(N_WFE)])

    S = sio.loadmat(m["figdata"], squeeze_me=True)
    resid = np.asarray(S["theta_true"], float) - np.asarray(S["theta_final"], float)
    d2 = float(resid @ np.linalg.pinv(cov) @ resid)

    # ---- Monte Carlo sweep over noise levels
    c_true = stacked(m, m["truth"])
    Bf = m["basis"].reshape(m["basis"].shape[0], -1)
    proj = Bf / np.einsum("ij,ij->i", Bf, Bf)[:, None]          # c = proj @ I
    npix = Bf.shape[1]
    sweep = []
    for s in SWEEP:
        errs, wfes, iters = [], [], []
        for _ in range(N_REAL[m["key"]]):
            c_obs = c_true + np.concatenate([proj @ rng.normal(0, s, npix) for _ in range(F)])
            th = m["truth"].copy()
            for it in range(12):
                step = A @ (c_obs - stacked(m, th))
                th = th + step
                if np.all(np.abs(step) < 1e-4 * m["steps"]):
                    break
            iters.append(it + 1)
            errs.append(m["truth"] - th)
            wfes.append(wfe(m["truth"] - th))
        errs = np.array(errs)
        sweep.append(dict(sigma=s, rms_err=np.sqrt((errs ** 2).mean(axis=0)),
                          lin_err=sig_theta * s / SIGMA, wfe_median=float(np.median(wfes)),
                          wfe_lin_median=float(np.median(wfe_lin)) * s / SIGMA,
                          max_iters=int(max(iters))))
        print(f"  [{m['key']}] sigma={s}: rms err {np.round(sweep[-1]['rms_err'], 4)} "
              f"(linear {np.round(sweep[-1]['lin_err'], 4)}), median WFE {sweep[-1]['wfe_median']:.3g} "
              f"(linear {sweep[-1]['wfe_lin_median']:.3g}), max iters {sweep[-1]['max_iters']}")

    return dict(J=J, lin=lin, N1=N1, cov=cov, sig_theta=sig_theta, floor=floor,
                chi2_noise=chi2_noise, ratio=ratio, sv=sv, wfe_lin=wfe_lin,
                resid=resid, d2=d2, obs_cost=float(np.asarray(S["objective"])[-1]),
                obs_wfe=float(S["wfe_final"]), n_coef=J.shape[0], sweep=sweep)


def report(m, r):
    print(f"\n===== {m['key']} =====")
    print(f"Jacobian step check (max rel. change, 2h vs h): {np.round(r['lin'], 4)}")
    sd = np.sqrt(np.diag(r["N1"]))
    print(f"coefficient noise sd per mode: {sd.min():.3g} .. {sd.max():.3g}; "
          f"tr N per field {np.trace(r['N1']):.3g}")
    print(f"N/Sigma diagonal ratio: median {np.median(r['ratio']):.2g}, max {r['ratio'].max():.2g}")
    print(f"expected -2 ln L shift at truth from noise, sum_f tr(Sigma^-1 N): {r['chi2_noise']:.3g}")
    print(f"expected DE cost floor {r['floor']:.3g} ({r['n_coef']} coefficients, "
          f"{len(m['truth'])} parameters); observed {r['obs_cost']:.3g}")
    print(f"singular values of J (step-scaled): {r['sv']:.3g}" if np.ndim(r['sv']) == 0 else
          f"singular values of J (step-scaled): {np.array2string(r['sv'], precision=3)}")
    for n, u, s, e in zip(m["names"], m["units"], r["sig_theta"], r["resid"]):
        print(f"  {n:4s} predicted sd {s:10.4g} {u:3s}  observed residual {e:10.4g}  z {e / s:6.2f}")
    print(f"Mahalanobis d^2 of observed residual: {r['d2']:.3g} (chi^2 with {len(m['truth'])} dof)")
    q = np.percentile(r["wfe_lin"], [16, 50, 84])
    print(f"linearised WFE: median {q[1]:.3g} waves (16-84%: {q[0]:.3g}-{q[2]:.3g}); "
          f"observed {r['obs_wfe']:.3g}")


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    results = {}
    for make in (hubble_model, tma_model):
        m = make()
        r = analyse(m, rng)
        report(m, r)
        results[m["key"]] = r
    flat = {}
    for k, r in results.items():
        for name, v in r.items():
            if name == "sweep":
                for field in ("sigma", "rms_err", "lin_err", "wfe_median", "wfe_lin_median", "max_iters"):
                    flat[f"{k}_sweep_{field}"] = np.array([row[field] for row in v])
            else:
                flat[f"{k}_{name}"] = np.asarray(v)
    np.savez(OUT, **flat)
    print(f"\nwrote {OUT}")
