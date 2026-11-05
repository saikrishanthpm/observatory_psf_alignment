"""Optiland port of the Pearl TMA ``telescopeCreation`` Batoid prescription.

Mirrors the Batoid4LOFT compound-optic layout in ``psfdyn.py`` / notebook snippets:
sequential surfaces, metres → millimetres, wavelength metres → micrometres.

Batoid                          Optiland
------                          --------
``defineCoordinate(posX=...)``  ``surfaces.add(dx=..., dy=..., rx=..., ...)``
``mirrorWithSpider(w, h)``      annulus minus three rotated rectangular arms
``pupilSize=6.46`` (m)          ``set_aperture(EPD, 6460)`` (mm)
``WL = 650e-9`` (m)             ``wavelengths.add(0.65)`` (µm)
"""

from __future__ import annotations

import math
from typing import Literal

import optiland.backend as be
from optiland import optic, physical_apertures

# ---------------------------------------------------------------------------
# Batoid prescription (metres unless noted)
# ---------------------------------------------------------------------------
_M_TO_MM = 1000.0
_UM_TO_MM = 1e-3  # µm → mm for decenter DOF
_ARCSEC_TO_DEG = 1.0 / 3600.0

PUPIL_DIAM_M = 6.46
BACK_DIST_M = 7.5  # Batoid compoundOptic backDist; unused in sequential build

M1_ROC_M = -16.256
M1_CONIC = -0.9953569
M1_INNER_D_M = 1.38
M1_OUTER_D_M = 6.42
SPIDER_WIDTH_M = 0.3
SPIDER_HEIGHT_M = 4.0

M2_ROC_M = -1.660575
M2_CONIC = -1.566503
M2_OUTER_D_M = 0.7147252
M2_Z_M = -7.40000

M3_ROC_M = -1.830517
M3_CONIC = -0.718069
M3_OUTER_D_M = 0.920733
M3_Z_M = 0.05002

M4_Z_M = -0.983629
M4_OUTER_D_M = 0.0991468

DET_Z_M = 0.237581
DET_OUTER_D_M = 0.9

DEFAULT_WL_UM = 0.65
DEFAULT_MAX_FIELD_DEG = 0.17


def wl_to_um(wl: float) -> float:
    """Convert wavelength to Optiland units (µm).

    Accepts metres (``650e-9``), micrometres (``0.65``), or nanometres (``650``).
    """
    wl = float(wl)
    if wl < 1e-3:
        return wl * 1e6
    if wl > 100.0:
        return wl / 1000.0
    return wl


def _rotated_rect_polygon_mm(
    cx_mm: float,
    cy_mm: float,
    width_mm: float,
    height_mm: float,
    theta_rad: float,
) -> tuple[list[float], list[float]]:
    """Vertices of a rectangle centred at (cx, cy), matching Batoid ObscRectangle."""
    hw = 0.5 * width_mm
    hh = 0.5 * height_mm
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    xs: list[float] = []
    ys: list[float] = []
    for x, y in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(c * x - s * y + cx_mm)
        ys.append(s * x + c * y + cy_mm)
    return xs, ys


def _m1_spider_aperture_mm(
    *,
    inner_d_mm: float = M1_INNER_D_M * _M_TO_MM,
    outer_d_mm: float = M1_OUTER_D_M * _M_TO_MM,
    spider_width_mm: float = SPIDER_WIDTH_M * _M_TO_MM,
    spider_height_mm: float = SPIDER_HEIGHT_M * _M_TO_MM,
) -> physical_apertures.DifferenceAperture:
    """Batoid ``obscMirrorWithSpider(width, height)`` — 3-arm spider on annulus."""
    annulus = physical_apertures.RadialAperture(
        r_min=0.5 * inner_d_mm,
        r_max=0.5 * outer_d_mm,
    )
    h = spider_height_mm
    w = spider_width_mm
    # Same arm layout as Batoid4LOFT.optic.obscMirrorWithSpider (incl. literal 4/2 in y).
    arms = [
        (0.5 * h, 0.0, math.pi / 2),
        (-0.5 * h * math.sin(math.pi / 6), -2.0 * math.cos(math.pi / 6), -math.pi / 6),
        (
            -0.5 * h * math.sin(5 * math.pi / 6),
            -2.0 * math.cos(5 * math.pi / 6),
            -5 * math.pi / 6,
        ),
    ]
    aperture = annulus
    for cx, cy, theta in arms:
        x, y = _rotated_rect_polygon_mm(cx, cy, w, h, theta)
        aperture = aperture - physical_apertures.PolygonAperture(x=x, y=y)
    return aperture


def _sequential_thicknesses_mm(
    *,
    m2_z_m: float = M2_Z_M,
    m3_z_m: float = M3_Z_M,
    m4_z_m: float = M4_Z_M,
    det_z_m: float = DET_Z_M,
) -> tuple[float, float, float, float]:
    """Edge thicknesses (mm) from M1 (z=0) through image, matching Batoid coordSys.z."""
    m1_z = 0.0
    return (
        (m2_z_m - m1_z) * _M_TO_MM,
        (m3_z_m - m2_z_m) * _M_TO_MM,
        (m4_z_m - m3_z_m) * _M_TO_MM,
        (det_z_m - m4_z_m) * _M_TO_MM,
    )


def telescopeCreation(
    WL: float = DEFAULT_WL_UM,
    *,
    pxo: float = 0.0,
    pyo: float = 0.0,
    pdx: float = 0.0,
    pdy: float = 0.0,
    pdz: float = 0.0,
    ptx: float = 0.0,
    pty: float = 0.0,
    sdx: float = 0.0,
    sdy: float = 0.0,
    sdz: float = 0.0,
    stx: float = 0.0,
    sty: float = 0.0,
    fields: list[tuple[float, float]] | None = None,
    max_field_deg: float = DEFAULT_MAX_FIELD_DEG,
    field_bias_deg: float = 0.0,
    field_bias_mode: Literal["coord_break", "field_angle", "none"] = "none",
    field_bias_axis: Literal["x", "y"] = "x",
    name: str = "Pearl",
) -> optic.Optic:
    """Build the Pearl TMA in Optiland (Batoid ``telescopeCreation`` equivalent).

    Parameters
    ----------
    WL
        Wavelength in µm, metres (``650e-9``), or nm (``650``).
    pxo, pyo
        M1 decenter in **metres** (Batoid ``defineCoordinate(posX=pxo, posY=pyo)``).
        Added to ``pdx``/``pdy`` (µm misalignment DOF).
    pdx, pdy, pdz, ptx, pty
        Primary mirror decenter (µm) and tilt (arcsec); same as ``psfdyn.py``.
    sdx, sdy, sdz, stx, sty
        Secondary mirror decenter (µm) and tilt (arcsec).
    fields
        Registered field points in degrees (sets ``max_field`` normalization).
    max_field_deg
        Default field extent when ``fields`` is omitted.
    field_bias_deg, field_bias_mode, field_bias_axis
        Optional Zemax-style field bias on object surface (see ``build_stp_tma``).
        Default ``"none"`` matches the bare Batoid snippet (no object tilt).
    """
    wl_um = wl_to_um(WL)
    thick12, thick23, thick34, thick4d = _sequential_thicknesses_mm()

    if fields is None:
        fields = [(0.0, 0.0), (0.0, max_field_deg), (0.0, -max_field_deg)]

    tel = optic.Optic(name=name)
    tel.set_aperture(aperture_type="EPD", value=PUPIL_DIAM_M * _M_TO_MM)
    tel.wavelengths.add(value=wl_um, is_primary=True)

    obj_kw: dict = dict(index=0, radius=be.inf, thickness=be.inf)
    if field_bias_mode == "coord_break" and field_bias_deg != 0.0:
        tilt = be.deg2rad(field_bias_deg)
        if field_bias_axis == "x":
            obj_kw["rx"] = tilt
        else:
            obj_kw["ry"] = tilt
    tel.surfaces.add(**obj_kw)

    # M1 — Batoid: Quadric + annulus + 3-arm spider; stop at entrance pupil
    m1_dx_mm = pxo * _M_TO_MM + pdx * _UM_TO_MM
    m1_dy_mm = pyo * _M_TO_MM + pdy * _UM_TO_MM
    tel.surfaces.add(
        index=1,
        radius=M1_ROC_M * _M_TO_MM,
        conic=M1_CONIC,
        thickness=thick12 + pdz * _UM_TO_MM,
        dx=m1_dx_mm,
        dy=m1_dy_mm,
        rx=ptx * _ARCSEC_TO_DEG,
        ry=pty * _ARCSEC_TO_DEG,
        material="mirror",
        aperture=_m1_spider_aperture_mm(),
        is_stop=True,
    )

    # M2
    tel.surfaces.add(
        index=2,
        radius=M2_ROC_M * _M_TO_MM,
        conic=M2_CONIC,
        thickness=thick23 + sdz * _UM_TO_MM,
        dx=sdx * _UM_TO_MM,
        dy=sdy * _UM_TO_MM,
        rx=stx * _ARCSEC_TO_DEG,
        ry=sty * _ARCSEC_TO_DEG,
        material="mirror",
        aperture=physical_apertures.RadialAperture(
            r_max=0.5 * M2_OUTER_D_M * _M_TO_MM
        ),
    )

    # M3 — circular clear aperture (Batoid outDiam)
    tel.surfaces.add(
        index=3,
        radius=M3_ROC_M * _M_TO_MM,
        conic=M3_CONIC,
        thickness=thick34,
        material="mirror",
        aperture=physical_apertures.RadialAperture(
            r_max=0.5 * M3_OUTER_D_M * _M_TO_MM
        ),
    )

    # M4 — fold flat
    tel.surfaces.add(
        index=4,
        radius=be.inf,
        thickness=thick4d,
        material="mirror",
        aperture=physical_apertures.RadialAperture(
            r_max=0.5 * M4_OUTER_D_M * _M_TO_MM
        ),
    )

    # Detector (Batoid plane at z = DET_Z_M)
    tel.surfaces.add(index=5)

    tel.fields.set_type(field_type="angle")
    for hx, hy in fields:
        tel.fields.add(x=hx, y=hy)

    return tel


if __name__ == "__main__":
    import numpy as np
    from optiland.psf import MMDFTPSF
    from optiland.wavefront import OPD

    be.set_backend("numpy")
    be.set_precision("float64")

    tel = telescopeCreation(650e-9, pxo=0.0, pyo=0.0)
    psf = MMDFTPSF(tel, field=(0.0, 0.0), wavelength=0.65, num_rays=64)
    rms = float(
        OPD(tel, field=(0, 0), wavelength=0.65, num_rays=32, strategy="centroid").rms()
    )
    print("Pearl (Batoid port):", tel.name)
    print("surfaces:", len(tel.surfaces))
    print("max_field (deg):", float(be.to_numpy(tel.fields.max_field)))
    print("PSF peak:", float(np.asarray(psf.psf).max()))
    print("OPD RMS (centroid, waves):", rms)
