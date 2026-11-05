"""Build poster.pptx: an editable PowerPoint version of poster.tex.

One 42 x 36 in slide. Text is native and editable, the two results tables
are native PowerPoint tables, and figures (with their overlays) are images
from pptx_assets/. Text boxes are sized by measuring the text with the
installed Arial metrics, so the layout is computed rather than hand-placed.

Run with a Python that has Pillow:  python make_pptx.py
Assets (see README lines at the bottom of this docstring):
  pearl.png, hubble.png, block.png  -- 300 dpi renders of the *_annotated.pdf
  psf_progression.png, logo.png     -- copies of ../figures/ images
  eq.png, qr.png                    -- rendered from eq.tex / qr.tex
"""
import os
import zipfile
from xml.sax.saxutils import escape

from PIL import Image, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "pptx_assets")
OUT = os.path.join(HERE, "poster.pptx")
FONTDIR = "/System/Library/Fonts/Supplemental/"

EMU = 914400
SW, SH = 42.0, 36.0


def emu(inches):
    return int(round(inches * EMU))


# ---------------------------------------------------------------- colours
# Theme slots: dk2 navy, lt2 paper, accent1 red, accent2 primary blue,
# accent3 secondary orange, accent4 cross purple, accent5 citation slate.
THEME_COLORS = dict(dk1="000000", lt1="FFFFFF", dk2="0C234B", lt2="F4F4F1",
                    accent1="AB0520", accent2="4C78A8", accent3="F58518",
                    accent4="B279A2", accent5="556581", accent6="2A78D6",
                    hlink="0563C1", folHlink="954F72")
NAVY, RED, WHITE, PAPER, BLACK = "tx2", "accent1", "bg1", "bg2", "tx1"
CITE = "accent5"
CAPTION = "#404040"     # leading '#' marks a literal sRGB colour
INSTITUTE = "#CED3DB"


def fill_xml(color):
    if color is None:
        return "<a:noFill/>"
    if color.startswith("#"):
        return f'<a:solidFill><a:srgbClr val="{color[1:]}"/></a:solidFill>'
    return f'<a:solidFill><a:schemeClr val="{color}"/></a:solidFill>'


# ---------------------------------------------------------------- text runs
# A run is (text, style) where style is a dict with optional keys:
#   b, i (bool)   m: math variable (Times New Roman italic)
#   sup, sub      c: colour (scheme name or '#RRGGBB')
def R(text, **style):
    return (text, style)


def M(text, **style):
    return (text, dict(style, m=True))


def cite(n):
    return R(f"[{n}]", c=CITE)


_fonts = {}


def _font(style, size):
    if style.get("m"):
        name = "Times New Roman Bold Italic.ttf" if style.get("b") else "Times New Roman Italic.ttf"
    else:
        b, i = bool(style.get("b")), bool(style.get("i"))
        name = {(False, False): "Arial.ttf", (True, False): "Arial Bold.ttf",
                (False, True): "Arial Italic.ttf", (True, True): "Arial Bold Italic.ttf"}[(b, i)]
    eff = size * (0.65 if (style.get("sup") or style.get("sub")) else 1.0)
    key = (name, round(eff * 20))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONTDIR + name, key[1])
    return _fonts[key]


def text_width_pt(chars, size):
    """chars: list of (char, style). Width in points."""
    w = 0.0
    seg, seg_style = "", None
    for ch, st in chars + [(None, None)]:
        if st is not seg_style and seg:
            w += _font(seg_style, size).getlength(seg) / 20.0
            seg = ""
        if ch is None:
            break
        seg_style = st
        seg += ch
    return w


def wrap_lines(runs, size, width_in):
    """Greedy word wrap. Returns the number of lines."""
    chars = [(ch, st) for text, st in runs for ch in text]
    words, cur = [], []
    for ch, st in chars:
        if ch == " ":
            if cur:
                words.append(cur)
            cur = []
        else:
            cur.append((ch, st))
    if cur:
        words.append(cur)
    limit = width_in * 72.0 * 0.97          # 3 % safety margin
    space = _font({}, size).getlength(" ") / 20.0
    lines, line_w = 1, 0.0
    for wd in words:
        ww = text_width_pt(wd, size)
        if line_w == 0.0:
            line_w = ww
        elif line_w + space + ww <= limit:
            line_w += space + ww
        else:
            lines += 1
            line_w = ww
    return lines


LINE = 1.2   # exact line spacing as a multiple of the font size


class Para:
    def __init__(self, runs, size=27, bullet=False, align="l", after=6,
                 color=BLACK, bold=False):
        self.runs = [r if isinstance(r, tuple) else R(r) for r in runs]
        if bold:
            self.runs = [(t, dict(s, b=True)) for t, s in self.runs]
        self.size, self.bullet, self.align = size, bullet, align
        self.after, self.color = after, color

    @property
    def indent_in(self):
        return 0.36 * self.size / 24 if self.bullet else 0.0

    def height_in(self, width_in):
        n = wrap_lines(self.runs, self.size, width_in - self.indent_in)
        return (n * LINE * self.size + self.after) / 72.0

    def xml(self):
        sz = int(self.size * 100)
        ppr = [f'<a:pPr algn="{self.align}"']
        if self.bullet:
            ind = emu(self.indent_in)
            ppr.append(f' marL="{ind}" indent="-{ind}"')
        ppr.append(">")
        ppr.append(f'<a:lnSpc><a:spcPts val="{int(LINE * self.size * 100)}"/></a:lnSpc>')
        ppr.append('<a:spcBef><a:spcPts val="0"/></a:spcBef>')
        ppr.append(f'<a:spcAft><a:spcPts val="{int(self.after * 100)}"/></a:spcAft>')
        if self.bullet:
            ppr.append('<a:buClr><a:schemeClr val="accent1"/></a:buClr>'
                       '<a:buSzPct val="60000"/><a:buFont typeface="Arial"/>'
                       '<a:buChar char="&#9658;"/>')
        else:
            ppr.append("<a:buNone/>")
        ppr.append("</a:pPr>")
        runs = "".join(run_xml(t, s, sz, self.color) for t, s in self.runs)
        return f"<a:p>{''.join(ppr)}{runs}<a:endParaRPr lang=\"en-US\" sz=\"{sz}\" dirty=\"0\"/></a:p>"


def run_xml(text, st, sz, default_color):
    attrs = f'lang="en-US" sz="{sz}"'
    if st.get("b"):
        attrs += ' b="1"'
    if st.get("i") or st.get("m"):
        attrs += ' i="1"'
    if st.get("sup"):
        attrs += ' baseline="30000"'
    if st.get("sub"):
        attrs += ' baseline="-25000"'
    face = "Times New Roman" if st.get("m") else "+mn-lt"
    color = st.get("c", default_color)
    return (f'<a:r><a:rPr {attrs} dirty="0">{fill_xml(color)}<a:latin typeface="{face}"/>'
            f'</a:rPr><a:t>{escape(text)}</a:t></a:r>')


def paras_height(paras, width_in):
    return sum(p.height_in(width_in) for p in paras)


# ---------------------------------------------------------------- shapes
class Slide:
    def __init__(self):
        self.shapes, self.images, self.next_id = [], [], 3

    def _id(self):
        self.next_id += 1
        return self.next_id

    def rect(self, name, x, y, w, h, fill, line=None, line_w=0.0, geom="rect"):
        ln = (f'<a:ln w="{emu(line_w)}">{fill_xml(line)}</a:ln>' if line
              else "<a:ln><a:noFill/></a:ln>")
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{self._id()}" name="{escape(name)}"/>'
            f'<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>'
            f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{fill_xml(fill)}{ln}</p:spPr></p:sp>')

    def text(self, name, x, y, w, h, paras, anchor="t"):
        body = "".join(p.xml() for p in paras)
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{self._id()}" name="{escape(name)}"/>'
            f'<p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr>'
            f'<a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="{anchor}">'
            f'<a:noAutofit/></a:bodyPr><a:lstStyle/>{body}</p:txBody></p:sp>')

    def image(self, name, fname, x, y, w, h, descr):
        rid = f"rId{len(self.images) + 2}"
        self.images.append((rid, fname))
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{self._id()}" name="{escape(name)}" descr="{escape(descr)}"/>'
            f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')

    def table(self, name, x, y, col_w, rows, size=20, row_h=0.42):
        """rows: list of rows; each row is a list of cells; a cell is
        dict(runs=[...], align='l'|'r'|'ctr', span=n, merged=bool,
        top=pt, bottom=pt)."""
        sz = int(size * 100)
        grid = "".join(f'<a:gridCol w="{emu(c)}"/>' for c in col_w)
        trs = []
        for row in rows:
            tcs = []
            for cell in row:
                if cell.get("merged"):
                    tcs.append('<a:tc hMerge="1"><a:txBody><a:bodyPr/><a:lstStyle/><a:p>'
                               f'<a:endParaRPr lang="en-US" sz="{sz}" dirty="0"/></a:p></a:txBody><a:tcPr/></a:tc>')
                    continue
                span = f' gridSpan="{cell["span"]}"' if cell.get("span", 1) > 1 else ""
                runs = "".join(run_xml(t, s, sz, BLACK) for t, s in
                               [r if isinstance(r, tuple) else R(r) for r in cell["runs"]])
                def ln(tag, pt):
                    if pt:
                        return (f'<a:{tag} w="{int(pt * 12700)}" cap="flat" cmpd="sng" algn="ctr">'
                                '<a:solidFill><a:srgbClr val="000000"/></a:solidFill>'
                                f'<a:prstDash val="solid"/></a:{tag}>')
                    return f'<a:{tag} w="12700"><a:noFill/></a:{tag}>'
                tcpr = (f'<a:tcPr marL="{emu(0.08)}" marR="{emu(0.08)}" marT="{emu(0.03)}" '
                        f'marB="{emu(0.03)}" anchor="ctr">'
                        f'{ln("lnL", 0)}{ln("lnR", 0)}{ln("lnT", cell.get("top", 0))}'
                        f'{ln("lnB", cell.get("bottom", 0))}<a:noFill/></a:tcPr>')
                tcs.append(f'<a:tc{span}><a:txBody><a:bodyPr/><a:lstStyle/><a:p>'
                           f'<a:pPr algn="{cell.get("align", "l")}"><a:buNone/></a:pPr>{runs}'
                           f'<a:endParaRPr lang="en-US" sz="{sz}" dirty="0"/></a:p></a:txBody>{tcpr}</a:tc>')
            trs.append(f'<a:tr h="{emu(row_h)}">{"".join(tcs)}</a:tr>')
        w, h = sum(col_w), row_h * len(rows)
        self.shapes.append(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{self._id()}" name="{escape(name)}"/>'
            f'<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/>'
            f'</p:nvGraphicFramePr><p:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/></p:xfrm><a:graphic>'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table">'
            f'<a:tbl><a:tblPr firstRow="1"/><a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl>'
            f'</a:graphicData></a:graphic></p:graphicFrame>')
        return h


def img_h(fname, w):
    with Image.open(os.path.join(ASSETS, fname)) as im:
        return w * im.height / im.width


# ---------------------------------------------------------------- blocks
PAD, TPAD, BORDER, GAP = 0.354, 0.315, 0.059, 0.315   # 9 mm, 8 mm, 1.5 mm, 8 mm
TITLE_PT = 43


class Block:
    """Collects content laid out top-to-bottom inside a poster block."""

    def __init__(self, slide, name, title_runs, x, y, w, accent=False, extra=0.0):
        self.s, self.name, self.x, self.y, self.w = slide, name, x, y, w
        self.inner_x, self.inner_w = x + PAD, w - 2 * PAD
        self.color = RED if accent else NAVY
        title = Para(title_runs, size=TITLE_PT, bold=True, color=WHITE, after=0)
        self.title_h = title.height_in(w - 2 * TPAD) + 2 * 0.16
        self.title = title
        self.items = []          # deferred drawing callbacks
        self.cur = y + self.title_h + PAD
        # extra: space added after each stacked element so that every column
        # ends at the bottom margin; gaps counts the elements it applies to.
        self.extra, self.gaps = extra, 0

    def stretch(self):
        self.cur += self.extra
        self.gaps += 1

    def paras(self, label, paras, x=None, w=None, advance=True):
        x = self.inner_x if x is None else x
        w = self.inner_w if w is None else w
        h = paras_height(paras, w)
        y = self.cur
        self.items.append(lambda: self.s.text(f"{self.name} - {label}", x, y, w, h, paras))
        if advance:
            self.cur += h
            self.stretch()
        return h

    def picture(self, label, fname, frac, descr, after=0.15):
        w = self.inner_w * frac
        h = img_h(fname, w)
        x, y = self.inner_x + (self.inner_w - w) / 2, self.cur
        self.items.append(lambda: self.s.image(f"{self.name} - {label}", fname, x, y, w, h, descr))
        self.cur += h + after
        self.stretch()

    def space(self, inches):
        self.cur += inches

    def finish(self):
        # drop the last element's stretch and paragraph spcAft
        bottom = self.cur - self.extra - 6 / 72.0 + PAD
        self.gaps -= 1
        h = bottom - self.y
        s = self.s
        s.rect(f"{self.name} - frame", self.x, self.y, self.w, h, WHITE, self.color, BORDER)
        s.rect(f"{self.name} - title bar", self.x, self.y, self.w, self.title_h, self.color)
        if self.color == NAVY:
            s.rect(f"{self.name} - title rule", self.x, self.y + self.title_h - 0.079,
                   self.w, 0.157, RED)
        s.text(f"{self.name} - title", self.x + TPAD, self.y, self.w - 2 * TPAD,
               self.title_h, [self.title], anchor="ctr")
        for draw in self.items:
            draw()
        return bottom


# ---------------------------------------------------------------- content
BODY, SMALL = 27, 22
ROW_H = 0.46        # table row height (in) for SMALL text


def build_slide(ex):
    """ex: extra space (in) after each stacked element, per column."""
    s = Slide()

    # Title bar: logo (left), QR code (right). Bar and rule come from the
    # layout. The title is an ordinary text box: Keynote does not render a
    # filled title placeholder reliably.
    s.text("Title", 9.0, 0.12, 24.0, 1.35,
           [Para(["O Mirror, Where Art Thou?"], size=80, bold=True, align="ctr", color=WHITE,
                 after=0)], anchor="ctr")
    logo_h = 1.654
    logo_w = logo_h * 940 / 213
    s.image("Wyant College logo", "logo.png", 0.984, 1.83 - logo_h / 2, logo_w, logo_h,
            "The University of Arizona Wyant College of Optical Sciences logo")
    qr_box = 2.559 + 2 * 0.276
    s.rect("QR code background", SW - 0.984 - qr_box, 0.276, qr_box, qr_box, WHITE)
    s.image("QR code", "qr.png", SW - 0.984 - qr_box + 0.276, 0.552, 2.559, 2.559,
            "QR code linking to Sai Krishanth P.M.'s LinkedIn profile")
    s.text("QR code label", SW - 0.984 - qr_box - 0.315 - 3.2, 1.831 - 0.55, 3.2, 1.1,
           [Para(["Connect on"], size=30, align="r", color=WHITE, after=0),
            Para(["LinkedIn"], size=30, align="r", color=WHITE, after=0)], anchor="ctr")
    s.text("Subtitle", 9.0, 1.45, 24.0, 0.75,
           [Para(["Position retrieval using nested sampling and global minimization"],
                 size=42, align="ctr", color=WHITE, after=0)])
    s.text("Authors", 9.0, 2.2, 24.0, 0.6,
           [Para([R("Sai Krishanth P.M."), R("1,2", sup=True), R(", Sanchit Sabhlok"), R("1", sup=True),
                  R(", Jaren N. Ashcraft"), R("3", sup=True), R(", Kyle Van Gorkom"), R("1", sup=True),
                  R(", Ewan S. Douglas"), R("1", sup=True)], size=34, align="ctr", color=WHITE, after=0)])
    s.text("Affiliations", 6.0, 2.85, 30.0, 0.5,
           [Para([R("1", sup=True), R("Steward Observatory and "), R("2", sup=True),
                  R("Wyant College of Optical Sciences, University of Arizona      "),
                  R("3", sup=True), R("UC Santa Barbara      Contact: saikrishanth@arizona.edu")],
                 size=26, align="ctr", color=INSTITUTE, after=0)])

    top = 4.33
    c1x, c1w = 0.70, 11.85
    c2x, c2w = c1x + c1w + 0.55, 15.80
    c3x, c3w = c2x + c2w + 0.55, 11.85
    bottoms = {}

    # ---- Column 1: Motivation
    b = Block(s, "Motivation", ["Motivation"], c1x, top, c1w, extra=ex["c1"])
    b.paras("question", [Para(["Can we tell where a telescope’s mirrors are from its images "
                               "alone?"], size=32, bold=True, color=RED)])
    b.paras("context", [Para(["A misaligned mirror distorts the point spread function (PSF) "
                              "differently across the field. Phase retrieval recovers the "
                              "wavefront, not the mirror positions."])])
    b.paras("what we did", [
        Para([R("What we did:", b=True), " infer rigid-body mirror positions ", R("directly", i=True),
              " from multi-field PSFs."]),
        Para(["Ray-traced ", R("Hubble", i=True), " (5 secondary-mirror DoF) and a compact TMA "
              "(10 primary + secondary DoF) in Optiland ", cite(1), "."], bullet=True),
        Para(["Compared multi-field PSF coefficients via a Gaussian likelihood."], bullet=True),
        Para(["Mapped the posterior with nested sampling, then refined it with differential "
              "evolution."], bullet=True),
        Para(["Checked each solution by the residual wavefront error."], bullet=True),
    ])
    y = b.finish() + GAP

    # ---- Column 1: Validating on Hubble
    b = Block(s, "Validating on Hubble", ["Validating on ", R("Hubble", i=True)], c1x, y, c1w,
              extra=ex["c1"])
    b.paras("setup", [
        Para(["Ritchey–Chrétien model (2.4 m pupil, ", M("λ"), " = 0.55 µm) with five "
              "secondary-mirror degrees of freedom: decenter (", M("sdx"), ", ", M("sdy"),
              "), despace ", M("sdz"), " and tilts (", M("stx"), ", ", M("sty"),
              "). Noise-free PSFs at 5 fields, 15 circular-Bessel modes, priors of ±200 µm "
              "and ±0.167°."])])
    b.paras("findings", [
        Para(["Nautilus (42 800 likelihood calls, ln ", M("Z"), " ≈ +107.5) tightly constrains "
              "both tilts and, more loosely, ", M("sdz"), "."], bullet=True),
        Para([M("sdx"), " and ", M("sdy"), " stay nearly flat across the ±200 µm prior."], bullet=True),
        Para(["DE starts from the median of the final 100 posterior samples, (131.13, 56.64, "
              "40.87, 0.12, −0.03), which alone cuts the on-axis OPD RMS from 2.26 to 0.12 "
              "waves."], bullet=True),
    ])
    b.picture("corner plot", "hubble.png", 1.0,
              "Nautilus posterior corner plot for the five Hubble secondary-mirror parameters, "
              "with a table of full-posterior medians, 68% intervals and axis spans", after=0.1)
    b.paras("caption", [Para(["Nautilus posterior for the five secondary-mirror parameters; the "
                              "table gives full-posterior medians and 68% intervals."],
                             size=SMALL, color=CAPTION)])
    b.paras("closing", [Para(["Even so, DE using only the on-axis field recovers every "
                              "translation to within 0.01 µm of the truth (centre panel)."])])
    bottoms["c1"] = b.finish()

    # ---- Column 2: Inference pipeline
    b = Block(s, "Inference pipeline", ["Inference pipeline"], c2x, top, c2w, extra=ex["c2"])
    b.picture("diagram", "block.png", 1.0,
              "Inference pipeline flowchart: simulate PSFs with Optiland at each field point, "
              "project onto basis modes, build Gaussian log-likelihood, run Nautilus, take "
              "posterior median and bounds, run differential evolution, final residual WFE "
              "check; the covariance matrices are computed once before sampling", after=0.2)
    # Two side-by-side columns; the stretch is applied once, after both.
    lw, rw = b.inner_w * 0.49, b.inner_w * 0.47
    rx = b.inner_x + b.inner_w - rw
    y0 = b.cur
    hl = b.paras("likelihood intro", [Para(["Optiland ", cite(1),
                                            " PSF coefficients enter a Gaussian likelihood:"],
                                           after=4)], w=lw, advance=False)
    eq_w = 5.65
    eq_h = img_h("eq.png", eq_w)
    ey = y0 + hl
    b.items.append(lambda: s.image("Inference pipeline - likelihood equation", "eq.png",
                                   b.inner_x + (lw - eq_w) / 2, ey, eq_w, eq_h,
                                   "ln L(theta) = sum over fields f of -1/2 delta c_f transpose "
                                   "Sigma_f inverse delta c_f + const, with delta c_f = "
                                   "c_f(theta) - c_f,obs"))
    b.cur = ey + eq_h + 0.08
    hc = b.paras("covariance note", [Para([M("Σ", b=True), M("f", sub=True),
                                           " from 10", R("4", sup=True),
                                           " random misalignments."])], w=lw, advance=False)
    left_end = b.cur + hc
    b.cur = y0
    hr = b.paras("steps", [
        Para([R("Nested sampling", b=True), " (Nautilus ", cite("2,3"), ") maps the posterior."],
             bullet=True, after=8),
        Para([R("Differential evolution", b=True), " ", cite("4,5"),
              " refines the Nautilus starting point."], bullet=True, after=8),
        Para([R("Validation:", b=True), " on-axis OPD RMS of the residual telescope."], bullet=True),
    ], x=rx, w=rw, advance=False)
    b.cur = max(y0 + hr, left_end)
    b.stretch()
    y = b.finish() + GAP

    # ---- Column 2: feature block
    b = Block(s, "Hubble result", ["Algorithm recovers ", R("Hubble", i=True),
                                   "’s optical alignment to numerical precision"],
              c2x, y, c2w, accent=True, extra=ex["c2"])
    b.picture("convergence figure", "psf_progression.png", 1.0,
              "Hubble on-axis PSF and RMS wavefront error as differential evolution refines the "
              "alignment, from 2.26 waves uncorrected to 2.6e-8 waves, with each "
              "secondary-mirror parameter converging to the injected truth", after=0.2)
    lw, rw = b.inner_w * 0.55, b.inner_w * 0.42
    rx = b.inner_x + b.inner_w - rw
    y0 = b.cur
    hl = b.paras("panel caption", [Para([
        R("(a)", b=True), " On-axis PSF, uncorrected (2.26 waves) to final DE solution "
        "(2.6 × 10", R("−8", sup=True), " waves). ", R("(b)", b=True),
        " RMS wavefront error per DE generation. ", R("(c)", b=True),
        " Parameters converging to the injected truth."])], w=lw, advance=False)
    hr = b.paras("table lead", [Para(["Translations recovered to ",
                                      R("< 0.01 µm", b=True, c=RED), ":"], after=6)],
                 x=rx, w=rw, advance=False)
    hub_rows = [
        [dict(runs=[""], top=1.5, bottom=0.75),
         dict(runs=[R("Truth", b=True)], align="r", top=1.5, bottom=0.75),
         dict(runs=[R("Recovered (DE)", b=True)], align="r", top=1.5, bottom=0.75)],
    ]
    for var, unit, t, rcv in [("sdx", "µm", "52.22", "52.22"), ("sdy", "µm", "−113.86", "−113.86"),
                              ("sdz", "µm", "38.99", "38.99"), ("stx", "deg", "0.13", "0.13"),
                              ("sty", "deg", "−0.04", "−0.04")]:
        last = var == "sty"
        hub_rows.append([dict(runs=[M(var), f" ({unit})"], bottom=1.5 if last else 0),
                         dict(runs=[t], align="r", bottom=1.5 if last else 0),
                         dict(runs=[rcv], align="r", bottom=1.5 if last else 0)])
    cols = [1.85, 1.6, 2.55]
    tx = rx + (rw - sum(cols)) / 2
    ty = y0 + hr
    th = ROW_H * len(hub_rows)
    b.items.append(lambda: s.table("Hubble result - truth vs recovered table", tx, ty, cols,
                                   hub_rows, size=SMALL, row_h=ROW_H))
    b.cur = max(ty + th, y0 + hl) + 0.06
    b.stretch()
    y = b.finish() + GAP

    # ---- Column 2: References
    b = Block(s, "References", ["References"], c2x, y, c2w, extra=ex["c2"])
    refs = ["Harrison 2026, Optiland v0.6.0", "Lange 2023, MNRAS 525, 3181",
            "Skilling 2006, Bayesian Analysis 1, 833",
            "Storn & Price 1997, J. Global Optim. 11, 341",
            "Virtanen et al. 2020, Nature Methods 17, 261",
            "Kim, Choi & Douglas 2023, Proc. SPIE 12677, 126770E"]
    half = b.inner_w / 2 - 0.15
    y0 = b.cur
    hl = b.paras("list 1", [Para([f"[{i + 1}] {r}"], size=SMALL, after=6)
                            for i, r in enumerate(refs[:3])], w=half, advance=False)
    hr = b.paras("list 2", [Para([f"[{i + 4}] {r}"], size=SMALL, after=6)
                            for i, r in enumerate(refs[3:])],
                 x=b.inner_x + b.inner_w / 2 + 0.15, w=half, advance=False)
    b.cur = y0 + max(hl, hr) + 0.05
    b.paras("thanks", [Para(["We thank T. Eifler, N. Franz, K. Derby and S. Blomquist."],
                            size=SMALL)])
    bottoms["c2"] = b.finish()

    # ---- Column 3: Validating on a TMA
    b = Block(s, "Validating on a TMA", ["Validating on a TMA"], c3x, top, c3w, extra=ex["c3"])
    b.paras("setup", [
        Para(["Compact TMA ", cite(6), " (6.47 m pupil, ", M("λ"), " = 0.65 µm) with ten degrees "
              "of freedom: decenter, despace and tilt of both the primary and the secondary. PSFs "
              "at 10 field points carry Gaussian noise (", M("σ"), " = 0.01 per pixel) and are "
              "projected onto 11 Zernike modes."])])
    b.paras("results", [
        Para(["Nautilus: 93 200 likelihood calls, ln ", M("Z"), " ≈ +484.7."], bullet=True),
        Para(["DE recovers all translations to ", R("≤ 17 µm", b=True, c=RED),
              " and reduces on-axis OPD RMS from 8.24 to 7 × 10", R("−3", sup=True), " waves."],
             bullet=True),
    ])
    tma_rows = [
        [dict(runs=[""], top=1.5),
         dict(runs=[R("Primary", b=True)], align="ctr", span=2, top=1.5, bottom=0.75),
         dict(merged=True),
         dict(runs=[R("Secondary", b=True)], align="ctr", span=2, top=1.5, bottom=0.75),
         dict(merged=True)],
        [dict(runs=[""], bottom=0.75)] + [dict(runs=[h], align="r", bottom=0.75)
                                          for h in ["Truth", "Recovered", "Truth", "Recovered"]],
    ]
    data = [("dx", "µm", ["−152.75", "−159.43", "−42.31", "−38.01"]),
            ("dy", "µm", ["316.91", "309.97", "923.33", "939.85"]),
            ("dz", "µm", ["26.63", "24.77", "−32.75", "−34.56"]),
            ("tx", "deg", [("2.98", "−5"), ("2.22", "−4"), "0.0269", "0.0272"]),
            ("ty", "deg", [("1.61", "−5"), ("6.80", "−5"), "0.0098", "0.0098"])]
    for var, unit, vals in data:
        last = var == "ty"
        row = [dict(runs=[M(var), f" ({unit})"], bottom=1.5 if last else 0)]
        for v in vals:
            runs = [f"{v[0]} × 10", R(v[1], sup=True)] if isinstance(v, tuple) else [v]
            row.append(dict(runs=runs, align="r", bottom=1.5 if last else 0))
        tma_rows.append(row)
    cols = [1.6, 2.05, 2.15, 1.6, 1.85]
    tx = b.inner_x + (b.inner_w - sum(cols)) / 2
    ty = b.cur
    b.items.append(lambda: s.table("Validating on a TMA - truth vs recovered table", tx, ty, cols,
                                   tma_rows, size=SMALL, row_h=ROW_H))
    b.cur += ROW_H * len(tma_rows) + 0.15
    b.stretch()
    b.picture("corner plot", "pearl.png", 1.0,
              "Nautilus posterior corner plot for the ten TMA parameters, shaded by mirror, with "
              "the strongest primary-secondary degeneracies outlined and a table of medians, "
              "68% intervals and axis spans", after=0.1)
    b.paras("caption", [Para(["Nautilus posterior for the ten TMA parameters, shaded by mirror. "
                              "Outlined panels are the strongest primary–secondary "
                              "degeneracies."], size=SMALL, color=CAPTION)])
    b.paras("notes", [
        Para([M("pdx"), " and both primary tilts are nearly flat across their priors; only the "
              "secondary tilts are tightly constrained."], bullet=True),
        Para(["Degeneracies: ", M("pdz"), "–", M("sdz"), " trade off in focus; ", M("ptx"), "–",
              M("sdy"), " and ", M("pty"), "–", M("sdx"), " trade off in coma, because a tilt of "
              "one mirror and a decenter of the other both add nearly field-constant coma."],
             bullet=True),
    ])
    y = b.finish() + GAP

    # ---- Column 3: Conclusions
    b = Block(s, "Conclusions", ["Conclusions and next steps"], c3x, y, c3w, extra=ex["c3"])
    b.paras("bullets", [
        Para(["Mirror alignment is recovered directly from multi-field PSF coefficients."],
             bullet=True),
        Para([R("Hubble", b=True, i=True), R(":", b=True), " OPD RMS 2.26 → ",
              R("3 × 10", b=True, c=RED), R("−8", b=True, c=RED, sup=True), " waves."], bullet=True),
        Para([R("TMA:", b=True), " OPD RMS 8.24 → ", R("7 × 10", b=True, c=RED),
              R("−3", b=True, c=RED, sup=True), " waves."], bullet=True),
        Para(["Degeneracies motivate multi-field sensing."], bullet=True),
        Para(["Next: chromatic and realistic noise models, and laboratory and on-sky PSF data "
              "for future large space telescopes."], bullet=True),
    ])
    bottoms["c3"] = b.finish()
    return s, bottoms


# ---------------------------------------------------------------- package
NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
HDR = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
EMPTY_GRP = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
             '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
             '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>')


def rels(items):
    body = "".join(f'<Relationship Id="{i}" Type="{REL}/{t}" Target="{tg}"/>' for i, t, tg in items)
    return f'{HDR}<Relationships xmlns="{PKG_REL}">{body}</Relationships>'


def theme_xml():
    c = THEME_COLORS
    clr = "".join(
        (f'<a:{k}><a:srgbClr val="{c[k]}"/></a:{k}>') for k in
        ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5",
         "accent6", "hlink", "folHlink"])
    solid = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    line = lambda w: (f'<a:ln w="{w}" cap="flat" cmpd="sng" algn="ctr">{solid}'
                      '<a:prstDash val="solid"/><a:miter lim="800000"/></a:ln>')
    eff = '<a:effectStyle><a:effectLst/></a:effectStyle>'
    return (f'{HDR}<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            f'name="O Mirror poster"><a:themeElements><a:clrScheme name="O Mirror poster">{clr}'
            '</a:clrScheme><a:fontScheme name="O Mirror poster">'
            '<a:majorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont>'
            '<a:minorFont><a:latin typeface="Arial"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont>'
            f'</a:fontScheme><a:fmtScheme name="O Mirror poster"><a:fillStyleLst>{solid * 3}'
            f'</a:fillStyleLst><a:lnStyleLst>{line(6350)}{line(12700)}{line(19050)}</a:lnStyleLst>'
            f'<a:effectStyleLst>{eff * 3}</a:effectStyleLst><a:bgFillStyleLst>{solid * 3}'
            '</a:bgFillStyleLst></a:fmtScheme></a:themeElements><a:objectDefaults/>'
            '<a:extraClrSchemeLst/></a:theme>')


def lvl1(sz, bold=False, color="tx1", font="+mn-lt", algn="l"):
    b = ' b="1"' if bold else ""
    return (f'<a:lvl1pPr algn="{algn}"><a:defRPr sz="{sz}"{b}><a:solidFill><a:schemeClr val="{color}"/>'
            f'</a:solidFill><a:latin typeface="{font}"/></a:defRPr></a:lvl1pPr>')


def title_ph(id_, with_geom):
    geom = (f'<a:xfrm><a:off x="{emu(9.0)}" y="{emu(0.12)}"/><a:ext cx="{emu(24.0)}" '
            f'cy="{emu(1.35)}"/></a:xfrm>' if with_geom else "")
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{id_}" name="Title"/><p:cNvSpPr><a:spLocks noGrp="1"/>'
            f'</p:cNvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:spPr>{geom}</p:spPr>')


def master_xml():
    return (f'{HDR}<p:sldMaster {NS}><p:cSld><p:spTree>{EMPTY_GRP}{title_ph(2, True)}'
            '<p:txBody><a:bodyPr lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr"><a:noAutofit/>'
            '</a:bodyPr><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/><a:t>Poster title</a:t></a:r>'
            '</a:p></p:txBody></p:sp></p:spTree></p:cSld>'
            '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
            'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" '
            'folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/>'
            f'</p:sldLayoutIdLst><p:txStyles><p:titleStyle>{lvl1(8000, True, "bg1", "+mj-lt", "ctr")}'
            f'</p:titleStyle><p:bodyStyle>{lvl1(2400)}</p:bodyStyle>'
            f'<p:otherStyle>{lvl1(2400)}</p:otherStyle></p:txStyles></p:sldMaster>')


def layout_xml():
    bar = (f'<p:sp><p:nvSpPr><p:cNvPr id="3" name="Title bar"/><p:cNvSpPr/><p:nvPr userDrawn="1"/>'
           f'</p:nvSpPr><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{emu(SW)}" cy="{emu(3.6)}"/>'
           '</a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill>'
           '<a:schemeClr val="tx2"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>')
    rule = (f'<p:sp><p:nvSpPr><p:cNvPr id="4" name="Title rule"/><p:cNvSpPr/><p:nvPr userDrawn="1"/>'
            f'</p:nvSpPr><p:spPr><a:xfrm><a:off x="0" y="{emu(3.6)}"/><a:ext cx="{emu(SW)}" '
            f'cy="{emu(0.35)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
            '<a:solidFill><a:schemeClr val="accent1"/></a:solidFill><a:ln><a:noFill/></a:ln>'
            '</p:spPr></p:sp>')
    title = (f'{title_ph(2, True)}<p:txBody><a:bodyPr lIns="0" tIns="0" rIns="0" bIns="0" '
             'anchor="ctr"><a:noAutofit/></a:bodyPr><a:lstStyle>'
             f'{lvl1(8000, True, "bg1", "+mj-lt", "ctr")}</a:lstStyle><a:p><a:r><a:rPr lang="en-US"/>'
             '<a:t>Poster title</a:t></a:r></a:p></p:txBody></p:sp>')
    return (f'{HDR}<p:sldLayout {NS} type="titleOnly" preserve="1"><p:cSld name="Poster">'
            '<p:bg><p:bgPr><a:solidFill><a:schemeClr val="bg2"/></a:solidFill><a:effectLst/>'
            f'</p:bgPr></p:bg><p:spTree>{EMPTY_GRP}{bar}{rule}{title}</p:spTree></p:cSld>'
            '<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')


def slide_xml(s):
    return (f'{HDR}<p:sld {NS}><p:cSld><p:spTree>{EMPTY_GRP}{"".join(s.shapes)}'
            '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')


def write_package(s):
    ct = (f'{HDR}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/>'
          '<Default Extension="png" ContentType="image/png"/>'
          '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>'
          '<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
          '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>'
          '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>'
          '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>'
          '<Override PartName="/ppt/presProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"/>'
          '<Override PartName="/ppt/viewProps.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"/>'
          '<Override PartName="/ppt/tableStyles.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"/>'
          '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
          '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
          '</Types>')
    pres = (f'{HDR}<p:presentation {NS} saveSubsetFonts="1"><p:sldMasterIdLst>'
            '<p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>'
            f'<p:sldId id="256" r:id="rId2"/></p:sldIdLst><p:sldSz cx="{emu(SW)}" cy="{emu(SH)}"/>'
            '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
    core = (f'{HDR}<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>O Mirror, Where Art Thou?</dc:title><dc:creator>Sai Krishanth P.M.</dc:creator>'
            '</cp:coreProperties>')
    app = (f'{HDR}<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
           '<Slides>1</Slides></Properties>')
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", f'{HDR}<Relationships xmlns="{PKG_REL}">'
                   f'<Relationship Id="rId1" Type="{REL}/officeDocument" Target="ppt/presentation.xml"/>'
                   '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                   f'<Relationship Id="rId3" Type="{REL}/extended-properties" Target="docProps/app.xml"/>'
                   '</Relationships>')
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", pres)
        z.writestr("ppt/_rels/presentation.xml.rels", rels([
            ("rId1", "slideMaster", "slideMasters/slideMaster1.xml"),
            ("rId2", "slide", "slides/slide1.xml"),
            ("rId3", "presProps", "presProps.xml"),
            ("rId4", "viewProps", "viewProps.xml"),
            ("rId5", "theme", "theme/theme1.xml"),
            ("rId6", "tableStyles", "tableStyles.xml")]))
        z.writestr("ppt/presProps.xml", f'{HDR}<p:presentationPr {NS}/>')
        z.writestr("ppt/viewProps.xml", f'{HDR}<p:viewPr {NS}><p:gridSpacing cx="76200" cy="76200"/></p:viewPr>')
        z.writestr("ppt/tableStyles.xml", f'{HDR}<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                   'def="{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"/>')
        z.writestr("ppt/theme/theme1.xml", theme_xml())
        z.writestr("ppt/slideMasters/slideMaster1.xml", master_xml())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", rels([
            ("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml"),
            ("rId2", "theme", "../theme/theme1.xml")]))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout_xml())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", rels([
            ("rId1", "slideMaster", "../slideMasters/slideMaster1.xml")]))
        z.writestr("ppt/slides/slide1.xml", slide_xml(s))
        img_rels = [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml")]
        seen = {}
        for rid, fname in s.images:
            img_rels.append((rid, "image", f"../media/{fname}"))
            if fname not in seen:
                seen[fname] = True
                z.write(os.path.join(ASSETS, fname), f"ppt/media/{fname}")
        z.writestr("ppt/slides/_rels/slide1.xml.rels", rels(img_rels))


TARGET = SH - 0.75     # every column ends here (inner margin is 18 mm)

if __name__ == "__main__":
    # Column bottoms are linear in the per-column stretch, so two trial
    # builds give the stretch that puts every column on TARGET.
    _, b0 = build_slide(dict(c1=0.0, c2=0.0, c3=0.0))
    _, b1 = build_slide(dict(c1=0.1, c2=0.1, c3=0.1))
    ex = {c: max(0.0, (TARGET - b0[c]) * 0.1 / (b1[c] - b0[c])) for c in b0}
    print("stretch per element (in):", {c: round(v, 3) for c, v in ex.items()})
    slide, bottoms = build_slide(ex)
    write_package(slide)
    print("wrote", OUT)
    for k, v in bottoms.items():
        print(f"  {k} bottom = {v:.2f} in (page {SH} in, margin limit {SH - 0.7:.2f})")
