"""Build the STP TMA (Pearl) from CompoundOptic YAML in Optiland.

Source YAML: STP_TMA_Mark_12F+M1_Bending_SBTest_5_HC01 style prescription.
Units in the YAML are metres; Optiland uses millimetres for geometry and
micrometres for wavelength.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import optiland.backend as be
from optiland import optic, physical_apertures


# ---------------------------------------------------------------------------
# Prescription constants (metres in YAML -> mm internally)
# ---------------------------------------------------------------------------
_M_TO_MM = 1000.0
_UM_TO_MM = 1e-3  # alignment DOF: µm -> mm
_ARCSEC_TO_DEG = 1.0 / 3600.0

# Global / meta
EPD_MM = 6.46 * _M_TO_MM
WL_UM = 0.65
FIELD_BIAS_DEG = 0.17
# Default full-field extent (deg) used to set Optiland's max_field normalization.
# Must be > 0 or every OPD(field=...) traces on-axis (max_field * H == 0).
MAX_FIELD_DEG = 0.17

# Sequential gaps derived from coordSys.z (mirrors only; see build_stp_tma)
_THICK_M1_TO_M2_MM = -7.4 * _M_TO_MM
_THICK_M2_TO_M3_MM = (0.05 - (-7.4)) * _M_TO_MM
_THICK_M3_TO_M4_MM = (-0.9836285606999999 - 0.05) * _M_TO_MM
_THICK_M4_TO_IMAGE_MM = (0.2375813873 - (-0.9836285606999999)) * _M_TO_MM

# M1 — Sum(Asphere + Zernike); nominal coefs are all zero -> conic + even_asphere hooks
M1 = dict(
    radius_mm=-16.256 * _M_TO_MM,
    conic=-0.9953569,
    asphere_coefs=[0.0] * 8,  # imin=1 => r^4 .. r^18
    zernike_coefs=[0.0] * 200,
    zernike_norm_radius_mm=3.2503 * _M_TO_MM,
    aperture_inner_mm=0.65 * _M_TO_MM,
    aperture_outer_mm=3.2503 * _M_TO_MM,
)

M2 = dict(
    radius_mm=-1.660575 * _M_TO_MM,
    conic=-1.566503,
    clear_radius_mm=0.7147252 * _M_TO_MM,
    z_mm=-7.4 * _M_TO_MM,
)

M3 = dict(
    radius_mm=-1.830517 * _M_TO_MM,
    conic=-0.718069,
    rect_center_y_mm=0.24 * _M_TO_MM,
    rect_width_mm=0.7 * _M_TO_MM,
    rect_height_mm=0.46 * _M_TO_MM,
    z_mm=0.05 * _M_TO_MM,
)

M4 = dict(z_mm=-0.9836285606999999 * _M_TO_MM, clear_radius_mm=0.0991468 * _M_TO_MM)

DETECTOR = dict(
    z_mm=0.2375813873 * _M_TO_MM,
    y_mm=-0.2898788285 * _M_TO_MM,
    clear_radius_mm=1.568139e-05 * _M_TO_MM,
)


def _m1_aperture():
    return physical_apertures.RadialAperture(
        r_min=M1["aperture_inner_mm"],
        r_max=M1["aperture_outer_mm"],
    )


def _m3_aperture():
    cx = 0.0
    cy = M3["rect_center_y_mm"]
    hw = 0.5 * M3["rect_width_mm"]
    hh = 0.5 * M3["rect_height_mm"]
    return physical_apertures.RectangularAperture(
        x_min=cx - hw,
        x_max=cx + hw,
        y_min=cy - hh,
        y_max=cy + hh,
    )


def normalize_field(
    tel: optic.Optic,
    fx_deg: float,
    fy_deg: float = 0.0,
) -> tuple[float, float]:
    """Convert field angles in degrees to Optiland normalized (Hx, Hy).

    Optiland ``trace`` / ``OPD(field=...)`` expect normalized coordinates:
    ``angle_deg = H * tel.fields.max_field``.  ``max_field`` is the maximum
    radial extent of fields registered via ``tel.fields.add`` — if only (0, 0)
    is registered, ``max_field == 0`` and every ``field=`` argument is ignored.
    """
    mf = float(be.to_numpy(tel.fields.max_field))
    if mf == 0.0:
        raise ValueError(
            "tel.fields.max_field is 0; register a non-zero field extent "
            "(e.g. build_stp_tma(fields=[(0, 0), (0, max_deg), ...])) "
            "before passing degree-valued field angles to OPD/trace."
        )
    return fx_deg / mf, fy_deg / mf


def _m1_surface_type(asphere_coefs: list[float], zernike_coefs: list[float]) -> str:
    """Pick Optiland geometry for M1 (no native Sum surface)."""
    if any(c != 0.0 for c in zernike_coefs):
        return "zernike"
    if any(c != 0.0 for c in asphere_coefs):
        return "even_asphere"
    return "standard"


def build_stp_tma(
    *,
    wl: float = WL_UM,
    epd: float = EPD_MM,
    fields: list[tuple[float, float]] | None = None,
    field_bias_deg: float = FIELD_BIAS_DEG,
    max_field_deg: float = MAX_FIELD_DEG,
    field_bias_mode: Literal["coord_break", "field_angle"] = "coord_break",
    field_bias_axis: Literal["x", "y"] = "x",
    # rigid-body misalignments (same convention as tma_moredof.ipynb)
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
    m1_asphere_coefs: list[float] | None = None,
    m1_zernike_coefs: list[float] | None = None,
    name: str = "STP_TMA",
) -> optic.Optic:
    """Construct the STP three-mirror anastigmat + fold flat in Optiland.

    Parameters
    ----------
    wl
        Wavelength in micrometres (default 0.65).
    epd
        Entrance-pupil diameter in millimetres (default 6460).
    fields
        Field points registered on the prescription, in **degrees** (not normalized).
        They set ``tel.fields.max_field`` used to scale ``OPD(field=(Hx, Hy))`` and
        ``trace(Hx, Hy, ...)``.  Default: on-axis plus ±``max_field_deg``.
    field_bias_deg
        Field bias in degrees (YAML ``metaData.fieldBias``; Zemax COORDBRK PARM 3).
    max_field_deg
        Radial field extent in degrees for normalization when ``fields`` is omitted.
        Must be > 0.  Does not add extra ray angles by itself — only sets the scale
        for ``field=(Hx, Hy)`` passed to OPD/PSF (``angle = H * max_field``).
    field_bias_mode
        ``"coord_break"`` — Zemax-style: tilt object surface (PARM 3/4), field
        (0, 0) is on-axis in lens coordinates. ``"field_angle"`` — no object tilt;
        bias enters via ``fields`` as an angle in degrees.
    field_bias_axis
        ``"x"`` maps to Zemax COORDBRK tilt X (PARM 3); ``"y"`` to PARM 4.
    pdx, pdy, pdz, ptx, pty
        Primary mirror decenter (µm) and tilt (arcsec).
    sdx, sdy, sdz, stx, sty
        Secondary mirror decenter (µm) and tilt (arcsec).
    m1_asphere_coefs, m1_zernike_coefs
        Override M1 deformation coefficients from the YAML Sum surface.

    Returns
    -------
    optiland.optic.Optic
        Sequential prescription equivalent to the CompoundOptic YAML.
    """
    asphere = list(m1_asphere_coefs if m1_asphere_coefs is not None else M1["asphere_coefs"])
    zernike = list(m1_zernike_coefs if m1_zernike_coefs is not None else M1["zernike_coefs"])
    m1_type = _m1_surface_type(asphere, zernike)

    if fields is None:
        extent = max(max_field_deg, abs(field_bias_deg))
        # Register full field extent so max_field > 0 (required for OPD vs field).
        fields = [(0.0, 0.0), (0.0, extent), (0.0, -extent)]

    tel = optic.Optic(name=name)
    tel.set_aperture(aperture_type="EPD", value=epd)
    tel.wavelengths.add(value=wl)

    # Object — Zemax SURF 1 "Field Bias" COORDBRK (PARM 1/2 decenter, PARM 3/4/5 tilt)
    obj_kw: dict[str, Any] = dict(index=0, radius=be.inf, thickness=be.inf)
    if field_bias_mode == "coord_break" and field_bias_deg != 0.0:
        tilt = be.deg2rad(field_bias_deg)
        if field_bias_axis == "x":
            obj_kw["rx"] = tilt
        else:
            obj_kw["ry"] = tilt
    tel.surfaces.add(**obj_kw)

    # M1 (stop at primary; YAML stopSurface ENPP is the pupil reference plane)
    m1_kwargs: dict[str, Any] = dict(
        index=1,
        surface_type=m1_type,
        radius=M1["radius_mm"],
        conic=M1["conic"],
        thickness=_THICK_M1_TO_M2_MM + pdz * _UM_TO_MM,
        dx=pdx * _UM_TO_MM,
        dy=pdy * _UM_TO_MM,
        rx=ptx * _ARCSEC_TO_DEG,
        ry=pty * _ARCSEC_TO_DEG,
        material="mirror",
        aperture=_m1_aperture(),
        is_stop=True,
    )
    if m1_type == "even_asphere":
        m1_kwargs["coefficients"] = asphere
    elif m1_type == "zernike":
        m1_kwargs["coefficients"] = zernike
        m1_kwargs["norm_radius"] = M1["zernike_norm_radius_mm"]
        m1_kwargs["zernike_type"] = "fringe"
    tel.surfaces.add(**m1_kwargs)

    # M2
    tel.surfaces.add(
        index=2,
        radius=M2["radius_mm"],
        conic=M2["conic"],
        thickness=_THICK_M2_TO_M3_MM + sdz * _UM_TO_MM,
        dx=sdx * _UM_TO_MM,
        dy=sdy * _UM_TO_MM,
        rx=stx * _ARCSEC_TO_DEG,
        ry=sty * _ARCSEC_TO_DEG,
        material="mirror",
        aperture=physical_apertures.RadialAperture(r_max=M2["clear_radius_mm"]),
    )

    # M3 — rectangular clear region (YAML ClearRectangle)
    tel.surfaces.add(
        index=3,
        radius=M3["radius_mm"],
        conic=M3["conic"],
        thickness=_THICK_M3_TO_M4_MM,
        material="mirror",
        aperture=_m3_aperture(),
    )

    # M4 fold flat (YAML Exit Pupil plane is a separate non-sequential item; skipped)
    tel.surfaces.add(
        index=4,
        radius=be.inf,
        thickness=_THICK_M4_TO_IMAGE_MM,
        material="mirror",
        aperture=physical_apertures.RadialAperture(r_max=M4["clear_radius_mm"]),
    )

    # Image / detector (decenter matches YAML coordSys.y on Flat Detector)
    tel.surfaces.add(
        index=5,
        dy=DETECTOR["y_mm"],
    )

    tel.fields.set_type(field_type="angle")
    for hx, hy in fields:
        tel.fields.add(x=hx, y=hy)

    return tel


def build_stp_tma_from_yaml(path: str | Path, **kwargs: Any) -> optic.Optic:
    """Load a CompoundOptic YAML file and build the matching Optiland optic.

    Requires PyYAML. Only a subset of the YAML schema is interpreted; unknown
    keys fall back to the embedded constants in this module.
    """
    import yaml

    path = Path(path)
    data = yaml.safe_load(path.read_text())
    osys = data["opticalSystem"]
    meta = data.get("metaData", {})

    kwargs.setdefault("name", meta.get("file", path.stem))
    if "wavelengths" in meta and meta["wavelengths"]:
        kwargs.setdefault("wl", float(meta["wavelengths"][0]) * 1e6)  # m -> µm
    if "fieldBias" in meta:
        kwargs.setdefault("field_bias_deg", float(meta["fieldBias"]))
    if "fields" in meta:
        kwargs.setdefault("fields", [(float(f[0]), float(f[1])) for f in meta["fields"]])

    pupil_m = float(osys.get("pupilSize", EPD_MM / _M_TO_MM))
    kwargs.setdefault("epd", pupil_m * _M_TO_MM)

    items = {item["name"]: item for item in osys["items"]}
    m1 = items["M1"]["surface"]
    asp = next(i for i in m1["items"] if i["type"] == "Asphere")
    zern = next(i for i in m1["items"] if i["type"] == "Zernike")
    kwargs.setdefault("m1_asphere_coefs", list(asp.get("coefs", [])))
    kwargs.setdefault("m1_zernike_coefs", list(zern.get("coef", [])))

    return build_stp_tma(**kwargs)


if __name__ == "__main__":
    import numpy as np
    from optiland.psf import MMDFTPSF

    be.set_backend("numpy")
    be.set_precision("float64")

    tel = build_stp_tma()
    psf = MMDFTPSF(tel, field=(0.0, 0.0), wavelength=WL_UM, num_rays=128)
    peak = float(np.asarray(psf.psf).max())
    print("STP TMA built:", tel.name)
    print("surfaces:", len(tel.surfaces))
    print("PSF peak (on-axis, 128 rays):", peak)
