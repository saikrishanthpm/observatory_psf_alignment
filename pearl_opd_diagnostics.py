"""Compare Optiland OPD for pearl.zmx against typical Zemax wavefront settings.

Run:
    python pearl_opd_diagnostics.py

pearl.zmx is a *nominal* sequential Pearl TMA (no M1 spider, no field-bias COORDBRK).
``load_zemax_file("pearl.zmx")`` matches the Zemax geometry; use it instead of
``telescopeCreation`` / ``build_stp_tma`` when comparing to this file.
"""

from __future__ import annotations

import optiland.backend as be
from optiland.fileio import load_zemax_file
from optiland.wavefront import OPD

PEARL_ZMX = "pearl.zmx"
WL_UM = 0.65


def field_deg(tel, fx_deg: float, fy_deg: float = 0.0) -> tuple[float, float]:
    """Zemax field angles (deg) → Optiland normalized (Hx, Hy)."""
    mf = float(be.to_numpy(tel.fields.max_field))
    if mf == 0.0:
        raise ValueError("max_field is 0 on this optic")
    return fx_deg / mf, fy_deg / mf


def opd_rms_nm(
    tel,
    field_norm: tuple[float, float],
    *,
    strategy: str = "centroid",
    num_rays: int = 128,
) -> tuple[float, float]:
    opd = OPD(
        tel,
        field=field_norm,
        wavelength=WL_UM,
        num_rays=num_rays,
        strategy=strategy,
    )
    w = float(opd.rms())
    return w, w * WL_UM * 1000.0


def main() -> None:
    be.set_backend("numpy")
    be.set_precision("float64")

    tel = load_zemax_file(PEARL_ZMX)
    mf = float(be.to_numpy(tel.fields.max_field))
    zmx_fields_y = [float(f.y) for f in tel.fields.fields]

    print("=" * 60)
    print("pearl.zmx → Optiland")
    print("=" * 60)
    print(f"surfaces: {len(tel.surfaces)}  EPD (mm): {float(be.to_numpy(tel.paraxial.EPD())):.2f}")
    print(f"Zemax YFLN (deg): {zmx_fields_y}")
    print(f"max_field (deg): {mf}")
    print(f"ROPD 2 in .zmx = exit-pupil reference (Zemax default wavefront plane)")
    print()
    print("Optiland OPD is in WAVES.  nm = waves × λ_nm = waves × 650 at 0.65 µm")
    print()

    strategies = ("chief_ray", "centroid", "best_fit")
    print(f"{'field (deg)':>14}  {'strategy':>12}  {'waves':>10}  {'nm':>8}")
    print("-" * 52)

    for fy in [0.0, 0.055, 0.17, 0.285]:
        fn = field_deg(tel, 0.0, fy)
        for strat in strategies:
            w, nm = opd_rms_nm(tel, fn, strategy=strat)
            print(f"{fy:>14.3f}  {strat:>12}  {w:>10.6f}  {nm:>8.2f}")

    print()
    print("Typical Zemax ↔ Optiland pitfalls (order-of-magnitude gaps):")
    print("  1. Units: Zemax map often nm; Optiland .view() colorbar is waves.")
    print("  2. Reference: OPD() defaults to chief_ray; centroid ≈ 2× lower here;")
    print("     best_fit ≈ 10× lower on-axis. Match Zemax analysis type explicitly.")
    print("  3. Field: pass normalized H, not degrees:")
    print(f"     0.17° → field={field_deg(tel, 0, 0.17)}  (not (0, 0.17))")
    print("  4. Model: pearl.zmx has no M1 spider; torchnesty/build_stp_tma do.")
    print("  5. Use num_rays≥64; default 15 is sparse for annular pupils elsewhere.")
    print()
    print("On-axis match target for pearl.zmx (this import):")
    w, nm = opd_rms_nm(tel, (0, 0), strategy="chief_ray")
    print(f"  chief_ray  ~{w:.4f} waves  (~{nm:.1f} nm)  ← closest to Zemax WFE map")
    w, nm = opd_rms_nm(tel, (0, 0), strategy="centroid")
    print(f"  centroid   ~{w:.4f} waves  (~{nm:.1f} nm)")


if __name__ == "__main__":
    main()
