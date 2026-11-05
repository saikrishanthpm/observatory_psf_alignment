"""
export_hubble_noise_figdata.py -- figure data for the noisy Hubble DE run
(hubbletilus_noise.ipynb).

Uses all 2000 DE generations logged in solutions_log_noise.txt (the whole run,
maxiter=2000; each record wraps onto two lines), computes RMS WFE at every generation and the
PSFs for initial / Nautilus / snapshots / final / aligned, and writes
hubble_noise_figdata.mat for make_tma_progression_figure.py:

    python export_hubble_noise_figdata.py
    python make_tma_progression_figure.py hubble_noise_figdata.mat Masters_Thesis/figures/psf_progression

The Hubble model is taken from the notebook's `def hubble` cell, so the figure
uses exactly the optic the run used. Truth and the Nautilus starting point (x0)
are copied from the notebook's outputs; the final parameters are the best
vector logged at generation 2000, which L-BFGS-B polishing left unchanged.
"""
import json
import os
import tempfile

import numpy as np
from optiland import optic, physical_apertures

from export_psf_figure_data import _RECORD, export_psf_figure_data, parse_de_log

NB = "hubbletilus_noise.ipynb"
LOG = "solutions_log_noise.txt"
N_GEN = 2000
OUT = "hubble_noise_figdata.mat"
NAMES = ["sdx", "sdy", "sdz", "stx", "sty"]

TRUTH = [52.22, -113.86, 38.99, 0.13, -0.04]   # cell 10
X0 = [51.06, -0.66, 39.6, 0.12, -0.04]         # cell 17, Nautilus median


def load_hubble(path=NB):
    cells = json.load(open(path))["cells"]
    src = next("".join(c["source"]) for c in cells
               if c["cell_type"] == "code" and "".join(c["source"]).startswith("def hubble("))
    ns = {"np": np, "optic": optic, "physical_apertures": physical_apertures}
    exec(src, ns)
    return ns["hubble"]


if __name__ == "__main__":
    hubble = load_hubble()

    # Use the first N_GEN records, so a later run appended to the log is ignored.
    with open(LOG) as f:
        text = f.read()
    recs = list(_RECORD.finditer(text))
    if len(recs) < N_GEN:
        raise ValueError(f"{LOG} holds only {len(recs)} complete records")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
        tmp.write(text[:recs[N_GEN - 1].end()] + "\n")

    try:
        # Snapshots A-D. The noisy cost reaches its floor (~6.2e-5) by about
        # generation 30 and is flat after that, so levels spaced in log between
        # the first and last cost (export_tma_figdata.py) pick generation 17
        # twice. Use the generations of the largest steps down to the floor
        # instead: cost 4.2e-4, 1.5e-4, 8.1e-5 and 6.4e-5 (within 4% of it).
        cost = parse_de_log(tmp.name)[-1]["objective"]
        snap_gens = [10, 17, 21, 30]
        print("  snapshot costs:", [f"{cost[g - 1]:.3g}" for g in snap_gens],
              f"floor {cost[-1]:.3g}")

        export_psf_figure_data(
            make_optic=lambda th: hubble(0.55, *th),
            theta_true=TRUTH,
            theta_ns=X0,
            theta_final=None,              # last logged best vector
            log_path=tmp.name,
            run=-1,
            cost_label="DE cost (sum of squared circular-Bessel residuals, 5 fields)",
            prior_bounds=[(-200, 200)] * 3 + [(-0.167, 0.167)] * 2,
            out_path=OUT,
            param_names=NAMES,
            param_units=["um"] * 3 + ["deg"] * 2,
            wavelength=0.55,
            crop=(96, 160),
            noise_sigma=0.01,
            snap_gens=snap_gens,
            title=f"Hubble secondary mirror, 5 DOF, noisy ({LOG}, first {N_GEN} generations)",
        )
    finally:
        os.unlink(tmp.name)
