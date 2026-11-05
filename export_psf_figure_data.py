"""
export_psf_figure_data.py
=========================

Collect everything the MATLAB figure script (make_psf_alignment_figure.m)
needs from a finished Nautilus -> differential-evolution (DE) alignment run
and write it to one .mat file.

Run it at the end of the notebook, while the kernel still holds the truth
vector, the Nautilus starting point (x0) and the DE result. It needs optiland,
numpy and scipy only.

What it computes
----------------
* Parses the DE callback log robustly. numpy wraps long arrays across lines,
  so `pd.read_csv(..., sep=r'\\s+', on_bad_lines='skip')` silently drops or
  shifts rows, and column 5 is not always the same parameter. The regex parser
  here reads every record.
* Warns if the log holds more than one run. The callbacks open the log in
  append mode ("a"), so re-running DE stacks runs in the same file. The last
  run is used by default.
* RMS WFE (waves, on-axis) at *every* DE generation, using the notebook's
  definition: nanstd(OPD_nominal - OPD_residual), where the residual optic has
  misalignment (theta_true - estimate).
* PSFs for: observed/initial, after Nautilus, snapshot generations, final, and
  the aligned (ideal) system. PSFs are stored as I / I_peak,ideal (optiland
  normalises MMDFTPSF so an ideal system peaks at 100), so a panel's peak value
  equals its Strehl ratio and every panel shares one intensity scale.
* Residual OPD maps (waves) for the same states.
* Pixels per lambda/D (from the working F/# and pixel pitch) for a scale bar.

Notebook cells to paste
-----------------------
hubbletilus.ipynb (Hubble, 5 secondary DOF), after the DE cell:

    from export_psf_figure_data import export_psf_figure_data
    export_psf_figure_data(
        make_optic=lambda th: hubble(0.55, *th),
        theta_true=[sdx_true, sdy_true, sdz_true, stx_true, sty_true],
        theta_ns=[float(v) for v in x0],
        theta_final=result.x,
        log_path="solutions_log.txt",
        out_path="hubbletilus_figdata.mat",
        param_names=["sdx", "sdy", "sdz", "stx", "sty"],
        param_units=["um", "um", "um", "deg", "deg"],
        wavelength=0.55,
        crop=(96, 160),
        title="Hubble secondary mirror, 5 DOF",
    )

tma_moredof.ipynb (Pearl TMA, 10 DOF, noisy PSFs), after the DE cell:

    from export_psf_figure_data import export_psf_figure_data
    names = ["pdx", "pdy", "pdz", "sdx", "sdy", "sdz", "ptx", "pty", "stx", "sty"]
    export_psf_figure_data(
        # keywords, because telescopeCreation's positional order
        # (WL, pdx, pdy, pdz, ptx, pty, sdx, ...) differs from theta's order
        make_optic=lambda th: telescopeCreation(0.65, **dict(zip(names, th))),
        theta_true=[pdx_true, pdy_true, pdz_true, sdx_true, sdy_true, sdz_true,
                    ptx_true, pty_true, stx_true, sty_true],
        theta_ns=[float(v) for v in x0],
        theta_final=result.x,
        log_path="solutions_logp.txt",          # whichever file this run wrote
        out_path="tma_figdata.mat",
        param_names=names,
        param_units=["um"] * 6 + [""] * 4,      # fill in the tilt unit
        wavelength=0.65,
        psf_kwargs={"num_rays": 512},
        crop=(384, 640),
        noise_sigma=0.01,                       # observed PSF only
        # snap_gens=[6, 16, 56, 118],           # the generations in the current
        #                                       # paper figure (solutions_logp.txt)
        title="Pearl TMA, 10 DOF",
    )

tma_moredof2.ipynb (Pearl TMA, absolute-position model, tilts in degrees).
The DE run in cell 49 used disp=True and no callback, so only its cost trace
exists. This cell (run after cell 49, with the notebook saved) makes the
reduced figure: initial / Nautilus / final / aligned, the cost trace, and
per-parameter errors:

    from export_psf_figure_data import export_psf_figure_data, parse_de_disp
    names = ["pdx", "pdy", "pdz", "sdx", "sdy", "sdz", "ptx", "pty", "stx", "sty"]
    export_psf_figure_data(
        make_optic=lambda th: telescopeCreation(0.65, **dict(zip(names, th))),
        theta_true=[pdx_true, pdy_true, pdz_true, sdx_true, sdy_true, sdz_true,
                    ptx_true, pty_true, stx_true, sty_true],
        theta_ns=[float(v) for v in x0[0]],     # x0 is a list holding one tuple here
        theta_final=result.x,
        log_path=None,
        cost_trace=parse_de_disp("tma_moredof2.ipynb", cell=49),
        cost_label="DE cost (sum of squared Zernike residuals, 10 fields)",
        prior_bounds=[(-800, 800)] * 3 + [(-2500, 2500)] * 3 + [(-0.01, 0.01)] * 2 + [(-0.4, 0.4)] * 2,
        out_path="tma_moredof2_figdata.mat",
        param_names=names,
        param_units=["um"] * 6 + ["deg"] * 4,
        wavelength=0.65,
        psf_kwargs={"num_rays": 512},
        crop=(384, 640),
        noise_sigma=0.01,
        title="Pearl TMA (tma_moredof2), 10 DOF",
    )

For the full figure (WFE per generation, snapshots A-D, parameter traces),
re-run cell 49 with a logging callback writing to a *new* file:

    import os
    LOG = "solutions_moredof2_de.txt"
    if os.path.exists(LOG):
        os.remove(LOG)
    def log_cb(intermediate_result):
        with open(LOG, "a") as f:
            f.write(f"Objective: {intermediate_result.fun}, Best X: {intermediate_result.x}\n")
    result = differential_evolution(lstep, bounds, x0=x0, tol=1e-10, disp=True, callback=log_cb)

then call export_psf_figure_data as above with log_path=LOG (drop cost_trace).

Then in MATLAB:  make_psf_alignment_figure('tma_figdata.mat')
"""

from __future__ import annotations

import re
import warnings

import numpy as np
from scipy.io import savemat

_RECORD = re.compile(r"Objective:\s*([^,]+),\s*Best X:\s*\[([^\]]*)\]", re.S)


# --------------------------------------------------------------------------
# log parsing
# --------------------------------------------------------------------------
def parse_de_log(path):
    """Return a list of runs, each a dict with 'objective' (N,) and 'theta' (N, P).

    A new run is assumed wherever the best objective increases (DE's best is
    non-increasing within one run) or the parameter count changes.
    """
    with open(path) as f:
        text = f.read()
    recs = _RECORD.findall(text)
    if not recs:
        raise ValueError(f"No 'Objective: ..., Best X: [...]' records in {path}")

    obj = np.array([float(o) for o, _ in recs])
    rows = [np.array(x.split(), dtype=float) for _, x in recs]

    runs, start = [], 0
    for i in range(1, len(rows)):
        if obj[i] > obj[i - 1] * (1 + 1e-9) or rows[i].size != rows[i - 1].size:
            runs.append((start, i))
            start = i
    runs.append((start, len(rows)))

    return [
        {"objective": obj[a:b], "theta": np.vstack(rows[a:b])} for a, b in runs
    ]


_DISP = re.compile(r"differential_evolution step (\d+): f\(x\)= ?([0-9.eE+-]+)")


def parse_de_disp(source, cell=None):
    """Best cost per generation from differential_evolution(..., disp=True) output.

    source: the printed text, a text file holding it, or a notebook (.ipynb)
    together with `cell`, the index of the cell whose output holds it. Use this
    when a run was made without a logging callback: it gives the cost trace
    but no parameter vectors, so no per-generation WFE or snapshots.
    """
    import json
    import os

    if isinstance(source, str) and os.path.exists(source):
        if source.endswith(".ipynb"):
            if cell is None:
                raise ValueError("pass cell=<index> for a notebook")
            nb = json.load(open(source))
            outs = nb["cells"][cell].get("outputs", [])
            text = "".join("".join(o.get("text", "")) for o in outs if o.get("output_type") == "stream")
        else:
            text = open(source).read()
    else:
        text = str(source)
    steps = _DISP.findall(text)
    if not steps:
        raise ValueError("no 'differential_evolution step N: f(x)= ...' lines found")
    g = np.array([int(a) for a, _ in steps])
    if not np.array_equal(g, np.arange(1, g.size + 1)):
        warnings.warn("DE step numbers are not 1..N; more than one run in this output?")
    return np.array([float(b) for _, b in steps])


# --------------------------------------------------------------------------
# optics helpers
# --------------------------------------------------------------------------
def _to_numpy(x):
    try:
        import optiland.backend as be

        return np.asarray(be.to_numpy(x))
    except Exception:
        return np.asarray(x)


def _crop(img, crop):
    if crop is None:
        return img
    if len(crop) == 2:
        r0, r1 = c0, c1 = crop
    else:
        r0, r1, c0, c1 = crop
    return img[r0:r1, c0:c1]


def export_psf_figure_data(
    make_optic,
    theta_true,
    theta_ns,
    log_path,
    out_path,
    param_names,
    param_units,
    wavelength,
    *,
    theta_final=None,
    cost_trace=None,
    cost_label="DE cost",
    prior_bounds=None,
    snap_gens=None,
    snap_wfe=None,
    n_snap=4,
    field=(0, 0),
    psf_kwargs=None,
    opd_kwargs=None,
    crop=None,
    noise_sigma=0.0,
    seed=0,
    run=-1,
    stage1="Nautilus",
    stage2="Differential evolution",
    title="",
    verbose=True,
):
    """Write the figure data .mat. See the module docstring for usage.

    make_optic(theta_residual) -> optiland Optic with that misalignment.
    theta_* are *estimates* c; the displayed system is make_optic(theta_true - c).
    log_path: the DE callback log (per-generation best vectors). If None, pass
    theta_final and optionally cost_trace (e.g. from parse_de_disp); the figure
    then shows initial / Nautilus / final only and plots the cost trace.
    prior_bounds: [(lo, hi), ...] per parameter; used to put parameter errors
    on one scale (|error| / prior half-width) in the MATLAB figure.
    opd_kwargs: passed to optiland's OPD, e.g. {"remove_tilt": True} to quote
    WFE without tip/tilt (which only shifts the image). Default {} matches the
    notebooks' nanstd(OPD_nominal - OPD_residual), tip/tilt included.
    snap_gens: 1-based DE generations to show as snapshots A, B, ... If None,
    the first generation reaching each WFE level is used: the levels in
    `snap_wfe` (waves RMS) if given, else `n_snap` levels spaced evenly in
    log(WFE) between generation 1 and the last generation.
    """
    from optiland.psf import MMDFTPSF
    from optiland.wavefront import OPD

    psf_kwargs = dict(psf_kwargs or {})
    opd_kwargs = dict(opd_kwargs or {})
    theta_true = np.asarray(theta_true, dtype=float).ravel()
    theta_ns = np.asarray(theta_ns, dtype=float).ravel()
    P = theta_true.size
    if len(param_names) != P or len(param_units) != P:
        raise ValueError("param_names / param_units must have one entry per parameter")

    # ---- DE trace ---------------------------------------------------------
    n_runs = 0
    if log_path is not None:
        runs = parse_de_log(log_path)
        n_runs = len(runs)
        if n_runs > 1:
            warnings.warn(
                f"{log_path} holds {n_runs} DE runs (the callback appends). "
                f"Using run index {run}; lengths: {[len(r['objective']) for r in runs]}. "
                "Delete the log before each run to avoid this."
            )
        tr = runs[run]
        theta_trace, objective = tr["theta"], tr["objective"]
        if theta_trace.shape[1] != P:
            raise ValueError(
                f"log has {theta_trace.shape[1]} parameters, theta_true has {P}. "
                "Is this the right log file?"
            )
        if theta_final is None:
            theta_final = theta_trace[-1]
    else:
        if theta_final is None:
            raise ValueError("without log_path, pass theta_final (e.g. result.x)")
        theta_trace = None
        objective = np.asarray(cost_trace, float).ravel() if cost_trace is not None else np.zeros(0)
    theta_final = np.asarray(theta_final, float).ravel()
    n_gen = len(objective)
    gen = np.arange(1, n_gen + 1)

    # ---- WFE --------------------------------------------------------------
    def opd_map(theta_residual):
        z = OPD(make_optic(theta_residual), field=field, wavelength=wavelength,
                **opd_kwargs).generate_opd_map()["z"]
        return _to_numpy(z)

    z_nom = opd_map(np.zeros(P))
    cache = {}

    def residual_opd(c):
        key = tuple(np.round(c, 12))
        if key not in cache:
            cache[key] = z_nom - opd_map(theta_true - np.asarray(c))
        return cache[key]

    def wfe(c):
        return float(np.nanstd(residual_opd(c)))

    wfe_trace = np.zeros(0)
    if theta_trace is not None:
        wfe_trace = np.empty(n_gen)
        for k, c in enumerate(theta_trace):
            wfe_trace[k] = wfe(c)
            if verbose and (k % 100 == 0 or k == n_gen - 1):
                print(f"  WFE gen {k + 1:5d}/{n_gen}: {wfe_trace[k]:.4g} waves "
                      f"({len(cache)} unique optics evaluated)")

    # ---- snapshot generations ---------------------------------------------
    if theta_trace is None:
        if snap_gens:
            warnings.warn("snap_gens ignored: no parameter log, so no snapshot vectors")
        snap_gens = []
    elif snap_gens is None:
        if snap_wfe is None:
            lo, hi = max(wfe_trace.min(), 1e-300), wfe_trace[0]
            snap_wfe = np.geomspace(hi, lo, n_snap + 2)[1:-1]
        snap_gens = []
        for thr in snap_wfe:
            hit = np.flatnonzero(wfe_trace <= thr)
            if hit.size and (hit[0] + 1) not in snap_gens and hit[0] + 1 < n_gen:
                snap_gens.append(int(hit[0] + 1))
        if verbose:
            print(f"  auto snapshot generations for WFE <= "
                  f"{[float(f'{t:.3g}') for t in snap_wfe]}: {snap_gens}")
    snap_gens = np.asarray(snap_gens, dtype=int)
    if theta_trace is not None:
        if snap_gens.size == 0:
            raise ValueError("No snapshot generations; pass snap_gens explicitly.")
        if snap_gens.min() < 1 or snap_gens.max() > n_gen:
            raise ValueError(f"snap_gens must lie in 1..{n_gen}")
        theta_snap = theta_trace[snap_gens - 1]
    else:
        theta_snap = np.zeros((0, P))

    # ---- PSFs ---------------------------------------------------------------
    rng = np.random.default_rng(seed)

    def psf(theta_residual, noisy=False):
        p = MMDFTPSF(make_optic(theta_residual), field=field, wavelength=wavelength, **psf_kwargs)
        img = _to_numpy(p.psf).astype(float)
        if noisy and noise_sigma > 0:
            img = img + rng.normal(0.0, noise_sigma, img.shape)
        return _crop(img, crop) / 100.0, p

    psf_ideal, p_ideal = psf(np.zeros(P))
    psf_initial, _ = psf(theta_true, noisy=True)
    psf_ns, _ = psf(theta_true - theta_ns)
    psf_final, _ = psf(theta_true - theta_final)
    psf_snap = (np.stack([psf(theta_true - c)[0] for c in theta_snap], axis=-1)
                if len(theta_snap) else np.zeros(0))

    try:
        lamD_pix = float(wavelength * float(p_ideal._get_working_FNO()) / float(p_ideal.pixel_pitch))
    except Exception:
        lamD_pix = np.nan
        warnings.warn("Could not get pixel pitch / F#; the MATLAB scale bar will be skipped.")

    out = {
        "title": title,
        "stage1": stage1,
        "stage2": stage2,
        "wavelength_um": float(wavelength),
        "field": np.asarray(field, float),
        "noise_sigma": float(noise_sigma),
        "wfe_tilt_removed": float(bool(opd_kwargs.get("remove_tilt", False))),
        "lamD_pix": lamD_pix,
        "param_names": np.array(list(param_names), dtype=object),
        "param_units": np.array(list(param_units), dtype=object),
        "theta_true": theta_true,
        "theta_ns": theta_ns,
        "theta_final": theta_final,
        "gen": gen.astype(float),
        "objective": objective,
        "cost_label": cost_label,
        "has_trace": float(theta_trace is not None),
        "theta_trace": theta_trace if theta_trace is not None else np.zeros(0),
        "wfe_trace": wfe_trace,
        "wfe_initial": wfe(np.zeros(P)),
        "wfe_ns": wfe(theta_ns),
        "wfe_final": wfe(theta_final),
        "snap_gen": snap_gens.astype(float),
        "wfe_snap": np.array([wfe(c) for c in theta_snap]),
        "psf_ideal": psf_ideal,
        "psf_initial": psf_initial,
        "psf_ns": psf_ns,
        "psf_snap": psf_snap,  # H x W x K
        "psf_final": psf_final,
        "opd_initial": residual_opd(np.zeros(P)),
        "opd_ns": residual_opd(theta_ns),
        "opd_snap": (np.stack([residual_opd(c) for c in theta_snap], axis=-1)
                     if len(theta_snap) else np.zeros(0)),
        "opd_final": residual_opd(theta_final),
        "log_path": str(log_path or ""),
        "n_runs_in_log": float(n_runs),
    }
    if prior_bounds is not None:
        pb = np.asarray(prior_bounds, float).reshape(P, 2)
        out["prior_center"] = pb.mean(axis=1)
        out["prior_halfwidth"] = np.diff(pb, axis=1).ravel() / 2
    savemat(out_path, out, do_compression=True)
    if verbose:
        print(f"Wrote {out_path}: {n_gen} generations, "
              f"{'snapshots at ' + str(snap_gens.tolist()) if len(snap_gens) else 'no parameter trace'}, "
              f"WFE initial {out['wfe_initial']:.3g} -> NS {out['wfe_ns']:.3g} -> "
              f"final {out['wfe_final']:.3g} waves RMS")
    return out
