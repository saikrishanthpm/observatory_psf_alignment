"""
noise_moves_analysis.py -- numbers for the draft thesis section on known moves of the
Hubble secondary (thesis_drafts/noise_moves_section.tex). Linearized as in Chapter 5:
observations at theta_true + delta_k, J_k = dc/dtheta there, and

    Cov = (sum_k J_k^T J_k)^-1 (sum_k J_k^T N_all J_k) (sum_k J_k^T J_k)^-1.

Computes, for the nominal design and for nominal + tilts of +-0.1 deg and of +-0.05 deg about
x and y: predicted s.d., correlations, expected DE cost floor with its scatter, the Mahalanobis
distance of the residuals of hubbletilus_noise_moves.ipynb and hubbletilus_noise_moves_0p05.ipynb,
and the on-axis WFE by Monte Carlo (300 draws); the gain over the same number of exposures
without moving as a function of tilt-move size; and how far DE had converged by each generation.
The +-0.05 deg estimate is the best member after 1000 generations in its DE log; polishing it
with L-BFGS-B moves it by less than 1e-7.

    python thesis_drafts/noise_moves_analysis.py    -> thesis_drafts/noise_moves_analysis.npz
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import hubble_moves_pool as hm                        # noqa: E402  (forward model of the notebook)
from optiland.wavefront import OPD                    # noqa: E402

TRUTH = np.array([52.22, -113.86, 38.99, 0.13, -0.04])
STEPS = np.array([0.2, 0.2, 0.2, 2e-5, 2e-5])         # central-difference steps of Chapter 5
SIGMA = 0.01
TILT_MOVE = 0.1                                       # the simulated design (degrees)
SIZES = np.array([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                  0.11, 0.12, 0.13, 0.14, 0.15, 0.167])
N_MC = 300

B = hm.basis_vectors_cached(nterms=15, npix=64).reshape(15, -1)
G = B @ B.T
N = SIGMA**2 * G / np.outer(np.diag(G), np.diag(G))
N_ALL = np.kron(np.eye(len(hm.FIELDS)), N)
_jac = {}


def jac(delta):
    key = tuple(np.round(delta, 9))
    if key not in _jac:
        t0 = TRUTH + np.asarray(delta, float)
        cols = []
        for k, h in enumerate(STEPS):
            e = np.zeros(5)
            e[k] = h
            cols.append((hm.position_coeffs(t0 + e) - hm.position_coeffs(t0 - e)).ravel() / (2 * h))
        _jac[key] = np.array(cols).T
    return _jac[key]


def tilt_design(dt):
    d = [np.zeros(5)]
    for k in (3, 4):
        for s in (1, -1):
            e = np.zeros(5)
            e[k] = s * dt
            d.append(e)
    return d


def linear(deltas):
    """Covariance and expected DE cost floor (mean, s.d.) for observations at TRUTH + deltas."""
    J = np.vstack([jac(d) for d in deltas])
    Nb = np.kron(np.eye(len(deltas)), N_ALL)
    inv = np.linalg.inv(J.T @ J)
    A = inv @ J.T
    cov = A @ Nb @ A.T
    M = (np.eye(len(J)) - J @ A) @ Nb
    return cov, np.trace(M), np.sqrt(2 * np.trace(M @ M))


def wfe(delta):
    """On-axis RMS OPD (waves) left by a residual misalignment delta, as in the thesis."""
    ref = OPD(hm.hubble(), field=(0, 0), wavelength=0.55).generate_opd_map()["z"]
    opd = OPD(hm.hubble(0.55, *delta), field=(0, 0), wavelength=0.55).generate_opd_map()["z"]
    return np.nanstd(ref - opd)


def de_log(name):
    """Best cost and member of every generation from a DE log written by the notebook callback."""
    lines = open(os.path.join(REPO, name)).read().splitlines()
    cost = np.array([float(l.split(", Best X: ")[0].removeprefix("Objective: ")) for l in lines])
    x = np.array([np.array(l.split(", Best X: ")[1].split(), dtype=float) for l in lines])
    return cost, x


def main():
    out = {}
    designs = {"nominal": [np.zeros(5)], "moves": tilt_design(TILT_MOVE), "moves_0p05": tilt_design(0.05)}
    rng = np.random.default_rng(1)
    for name, deltas in designs.items():
        cov, floor, floor_sd = linear(deltas)
        sd = np.sqrt(np.diag(cov))
        draws = rng.multivariate_normal(np.zeros(5), cov, N_MC)
        w = np.array([wfe(-d) for d in draws])
        out[f"{name}_cov"], out[f"{name}_sd"] = cov, sd
        out[f"{name}_floor"] = np.array([floor, floor_sd])
        out[f"{name}_wfe_pct"] = np.percentile(w, [16, 50, 84])
        corr = cov / np.outer(sd, sd)
        print(f"{name}: sd {np.array2string(sd, precision=4)}; r(sdy,stx) {corr[1, 3]:+.3f}, "
              f"r(sdx,sty) {corr[0, 4]:+.3f}; floor ({floor:.3e} +- {floor_sd:.1e}); "
              f"WFE 16/50/84% {np.array2string(out[f'{name}_wfe_pct'], precision=5)}")

    # residuals (truth - estimate, as in the thesis) of the runs: hubbletilus_noise_moves.ipynb (DE stopped
    # after 137 generations, polished) and hubbletilus_noise_moves_0p05.ipynb (1000 generations)
    runs = {"moves": np.array([52.05989855, -113.8543314, 38.99430205, 0.1299900951, -0.04002215713]),
            "moves_0p05": de_log("solutions_log_noise_moves_0p05.txt")[1][-1]}
    for name, est in runs.items():
        res = TRUTH - est
        out[f"{name}_residual"] = res
        out[f"{name}_mahalanobis"] = res @ np.linalg.solve(out[f"{name}_cov"], res)
        out[f"{name}_run_wfe"] = wfe(res)
        print(f"{name} run: residual {np.array2string(res, precision=6)}, z "
              f"{np.array2string(res / out[f'{name}_sd'], precision=2)}, d2 {out[f'{name}_mahalanobis']:.2f}, "
              f"WFE {out[f'{name}_run_wfe']:.3g} waves")

    # gain over the same number of exposures without moving, versus tilt-move size
    rep = out["nominal_sd"] / np.sqrt(5)
    gains = np.array([rep / np.sqrt(np.diag(linear(tilt_design(dt))[0])) for dt in SIZES])
    out["sizes"], out["gains"] = SIZES, gains
    for dt, g in zip(SIZES, gains):
        print(f"  +-{dt:.3f} deg: gain {np.array2string(g, precision=2)}")

    # DE convergence of the two runs
    for name, log in (("de_cost", "solutions_log_noise_moves.txt"), ("de_cost_0p05", "solutions_log_noise_moves_0p05.txt")):
        cost = de_log(log)[0]
        out[name] = cost
        for f in (1.04, 1.01, 1.001):
            print(f"{log}: within {100 * (f - 1):.1f}% of its final cost {cost[-1]:.4e} by generation "
                  f"{np.argmax(cost <= f * cost[-1]) + 1} of {len(cost)}")
    np.savez(os.path.join(HERE, "noise_moves_analysis.npz"), **out)


if __name__ == "__main__":
    main()
