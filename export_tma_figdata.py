"""
export_tma_figdata.py -- figure data for the TMA DE run made by fig_gen_de.py.

Reads the per-generation log (solutions_logtma.txt), computes RMS WFE at every
generation and the PSFs for initial / Nautilus / snapshots / final / aligned,
and writes tma_fig_gen_de_figdata.mat for make_tma_progression_figure.py.
Truth and the starting point (Nautilus median) come from fig_gen_de.py; the
final parameters are the last logged best vector (result.x is not saved).

    python export_tma_figdata.py
"""
import numpy as np

import fig_gen_de as g
from export_psf_figure_data import export_psf_figure_data, parse_de_log

LOG = "solutions_logtma.txt"
OUT = "tma_fig_gen_de_figdata.mat"
NAMES = ["pdx", "pdy", "pdz", "sdx", "sdy", "sdz", "ptx", "pty", "stx", "sty"]

if __name__ == "__main__":
    truth = [getattr(g, f"{n}_true") for n in NAMES]

    # Snapshots A-D: the first generation at which the DE cost falls below each
    # of four levels spaced evenly in log between the first and last cost.
    # (On-axis WFE is not monotonic here -- DE minimises a noisy 10-field cost --
    # so picking snapshots by WFE level would select chance dips.)
    cost = parse_de_log(LOG)[-1]["objective"]
    levels = np.geomspace(cost[0], cost[-1], 6)[1:-1]
    snap_gens = [int(np.flatnonzero(cost <= lev)[0] + 1) for lev in levels]

    export_psf_figure_data(
        # keywords, because telescopeCreation's positional order differs from theta's
        make_optic=lambda th: g.telescopeCreation(0.65, **dict(zip(NAMES, th))),
        theta_true=truth,
        theta_ns=g.x0,                 # DE started from the Nautilus median
        theta_final=None,              # last logged best vector
        log_path=LOG,
        run=-1,
        cost_label="DE cost (sum of squared Zernike residuals, 10 fields)",
        prior_bounds=[(-800, 800)] * 3 + [(-2500, 2500)] * 3 + [(-0.01, 0.01)] * 2 + [(-0.4, 0.4)] * 2,
        out_path=OUT,
        param_names=NAMES,
        param_units=["um"] * 6 + ["deg"] * 4,
        wavelength=0.65,
        psf_kwargs={"num_rays": 512},
        crop=(384, 640),
        noise_sigma=0.01,
        snap_gens=snap_gens,
        title="Pearl TMA (fig_gen_de.py), 10 DOF",
    )
