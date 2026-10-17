"""Build docs/How_It_Works.pdf, a plain-English guide to the project.

    python docs/make_explainer_pdf.py

Every number in the guide is read from models/metadata.json and the trained models, so re-running this
after retraining keeps the guide accurate.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from bankruptcy.config import FIGURES_DIR  # noqa: E402
from bankruptcy.data import load_horizon, train_test  # noqa: E402
from bankruptcy.features import meaning  # noqa: E402
from bankruptcy.predict import BankruptcyPredictor  # noqa: E402
from bankruptcy.report import LEGACY_V1  # noqa: E402

OUT = ROOT / "docs" / "How_It_Works.pdf"
TMP = Path(tempfile.mkdtemp())

# ----------------------------------------------------------------------------- look & feel
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic")

INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#898781")
LINE = colors.HexColor("#d9d8d2")
BLUE = colors.HexColor("#2a78d6")
BLUE_BG = colors.HexColor("#eaf2fc")
RED = colors.HexColor("#e34948")
RED_BG = colors.HexColor("#fdecec")
GREEN = colors.HexColor("#0ca30c")
GREEN_BG = colors.HexColor("#e9f6e9")
AMBER = colors.HexColor("#fab219")
AMBER_BG = colors.HexColor("#fff5df")
PANEL = colors.HexColor("#f4f4f1")

W = A4[0] - 4 * cm   # usable width

S = {
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=26, leading=32, textColor=INK, spaceAfter=10),
    "subtitle": ParagraphStyle("subtitle", fontName="Sans", fontSize=13, leading=18, textColor=INK2, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="Sans-Bold", fontSize=18, leading=23, textColor=INK, spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=12.5, leading=17, textColor=INK, spaceBefore=10, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=10, leading=15, textColor=INK, spaceAfter=7),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=8.5, leading=12, textColor=INK2, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.8, leading=12, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8.8, leading=12, textColor=INK),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=9.5, leading=13.5, textColor=INK),
    "boxc": ParagraphStyle("boxc", fontName="Sans", fontSize=9, leading=12.5, textColor=INK, alignment=TA_CENTER),
    "tile_v": ParagraphStyle("tile_v", fontName="Sans-Bold", fontSize=20, leading=24, textColor=INK, alignment=TA_CENTER),
    "tile_l": ParagraphStyle("tile_l", fontName="Sans", fontSize=8.5, leading=11, textColor=INK2, alignment=TA_CENTER),
    "arrow": ParagraphStyle("arrow", fontName="Sans", fontSize=10, leading=10, textColor=MUTED, alignment=TA_CENTER),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="body"):
    return [Paragraph(f"•&nbsp;&nbsp;{t}", ParagraphStyle("b", parent=S[style], leftIndent=12,
                                                          firstLineIndent=-10, spaceAfter=3)) for t in items]


def callout(content, bg=BLUE_BG, edge=BLUE, width=W):
    items = content if isinstance(content, list) else [content]
    t = Table([[items]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return t


def table(rows, widths, header=True, zebra=False):
    data = [[c if not isinstance(c, str) else P(c, "cellb" if (header and i == 0) else "cell") for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), BLUE_BG))
    if zebra:
        style += [("BACKGROUND", (0, i), (-1, i), PANEL) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def tiles(items):
    cells = [[P(v, "tile_v"), P(label, "tile_l")] for v, label in items]
    t = Table([[Table([[c[0]], [c[1]]], colWidths=[W / len(items) - 10]) for c in cells]],
              colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("LINEAFTER", (0, 0), (-2, -1), 2, colors.white),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def flow(steps, width=W * 0.78, colors_=None):
    """Vertical chain of boxes joined by arrows. steps = [(title, text), ...]"""
    rows = []
    for i, (title, text) in enumerate(steps):
        rows.append([P(f"<b>{i + 1}. {title}</b><br/>{text}", "box")])
        if i < len(steps) - 1:
            rows.append([P("↓", "arrow")])
    t = Table(rows, colWidths=[width])
    style = [("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]
    for i in range(0, len(rows), 2):
        bg, edge = (colors_ or {}).get(i // 2, (BLUE_BG, BLUE))
        style += [("BACKGROUND", (0, i), (0, i), bg), ("LINEBEFORE", (0, i), (0, i), 3, edge),
                  ("LEFTPADDING", (0, i), (0, i), 10), ("TOPPADDING", (0, i), (0, i), 4),
                  ("BOTTOMPADDING", (0, i), (0, i), 4)]
    t.setStyle(TableStyle(style))
    t.hAlign = "CENTER"
    return t


def arrow(d, x1, y1, x2, y2, color=MUTED, w=1.2):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w))
    ang = np.arctan2(y2 - y1, x2 - x1)
    a1, a2 = ang + np.pi * 0.85, ang - np.pi * 0.85
    d.add(Polygon([x2, y2, x2 + 6 * np.cos(a1), y2 + 6 * np.sin(a1), x2 + 6 * np.cos(a2), y2 + 6 * np.sin(a2)],
                  fillColor=color, strokeColor=color))


def box(d, x, y, w, h, lines, fill=BLUE_BG, edge=BLUE, size=8, bold_first=True):
    d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=edge, strokeWidth=0.8, rx=4, ry=4))
    n = len(lines)
    for i, line in enumerate(lines):
        yy = y + h / 2 + (n / 2 - i - 0.8) * (size + 3)
        d.add(String(x + w / 2, yy, line, fontName="Sans-Bold" if (i == 0 and bold_first) else "Sans",
                     fontSize=size, fillColor=INK, textAnchor="middle"))


def img(path, width=W):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw)


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.2 * cm, "How the Bankruptcy Risk Predictor works")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------- data for the guide
pred = BankruptcyPredictor()
META = pred.metadata
H = {int(k): v for k, v in META["horizons"].items()}
HS = sorted(H)


def yl(h):
    return f"{h} year ahead" if h == 1 else f"{h} years ahead"


def avg(key, src="test"):
    return np.mean([H[h]["metrics"][src][key] for h in HS])


n_total = sum(H[h]["n_train"] + H[h]["n_test"] for h in HS)
n_bankrupt = sum(H[h]["n_train_bankrupt"] + H[h]["n_test_bankrupt"] for h in HS)
common = sorted(set.intersection(*[set(H[h]["features"]) for h in HS]), key=lambda f: int(f[4:]))

# Worked example: the 1-year model's distressed and healthy test companies
EX_H = 1
m1 = H[EX_H]
ex_bad = m1["examples"]["Distressed company"]["values"]
ex_good = m1["examples"]["Healthy company"]["values"]
exp_bad = pred.explain(EX_H, ex_bad)
p_bad = float(pred.predict(EX_H, ex_bad)["bankruptcy_probability"].iloc[0])
p_good = float(pred.predict(EX_H, ex_good)["bankruptcy_probability"].iloc[0])
base_bad = exp_bad.attrs["base"]
z_bad = base_bad + exp_bad["contribution"].sum()
z_good = pred.explain(EX_H, ex_good)["contribution"].sum() + pred.explain(EX_H, ex_good).attrs["base"]


def directions(h, feats):
    """What the training data shows: bankruptcy rate among the lowest vs highest 10% of each ratio.
    (Raw data, not SHAP: when ratios are correlated, one ratio's SHAP push can point the 'wrong' way.)"""
    X_tr, _, y_tr, _ = train_test(load_horizon(h))
    out = {}
    for f in feats:
        v, y = X_tr[f], y_tr
        ok = v.notna()
        lo_cut, hi_cut = v[ok].quantile([0.1, 0.9])
        lo, hi = y[ok & (v <= lo_cut)].mean(), y[ok & (v >= hi_cut)].mean()
        mid = y[ok & (v > lo_cut) & (v < hi_cut)].mean()
        nums = f"lowest tenth {lo:.0%} went bankrupt, middle {mid:.0%}, highest tenth {hi:.0%}"
        if lo > 2 * hi and lo > 1.5 * mid:
            out[f] = f"<b>Low values are the danger sign</b> ({nums})"
        elif hi > 2 * lo and hi > 1.5 * mid:
            out[f] = f"<b>High values are the danger sign</b> ({nums})"
        elif lo > 1.5 * mid and hi > 1.5 * mid:
            out[f] = f"<b>Both extremes are riskier</b> ({nums})"
        else:
            out[f] = f"Weak pattern on its own; useful in combination ({nums})"
    return out


# ----------------------------------------------------------------------------- charts made for the guide
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.family": "DejaVu Sans", "font.size": 9,
})


def chart_example():
    top = exp_bad.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    cols = ["#e34948" if v > 0 else "#2a78d6" for v in top["contribution"]]
    ax.barh(range(len(top)), top["contribution"], 0.6, color=cols)
    ax.set_yticks(range(len(top)), [f"{mn} ({f})" for mn, f in zip(top["meaning"], top["feature"])],
                  fontsize=8, color="#52514e")
    ax.axvline(0, color="#c3c2b7", lw=1)
    ax.set_xlabel("Push on the risk score")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    path = TMP / "example.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def chart_sigmoid():
    z = np.linspace(-8, 8, 300)
    cal = m1["calibration"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(z, 1 / (1 + np.exp(-z)), color="#2a78d6", lw=2)
    ax.axhline(m1["threshold"], color="#898781", lw=1)
    ax.text(-7.8, m1["threshold"] + 0.03, f"threshold {m1['threshold']:.0%}", color="#52514e", fontsize=8)
    for zz, pp, label, c in [(z_bad, p_bad, "distressed example", "#e34948"),
                             (z_good, p_good, "healthy example", "#0ca30c")]:
        zc = float(np.clip(zz, -8, 8))
        ax.plot([zc], [pp], "o", ms=8, color=c, mec="white", mew=2, zorder=5)
        ax.annotate(f"{label}: {pp:.1%}", (zc, pp), xytext=(10, -14) if pp > 0.5 else (0, 28),
                    ha="left" if pp > 0.5 else "center", textcoords="offset points", fontsize=8, color="#0b0b0b")
    ax.set_xlabel("Risk score (sum of all trees, after calibration)")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.03, 1.05)
    fig.tight_layout()
    path = TMP / "sigmoid.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- drawings
def drawing_pictogram():
    rate = n_bankrupt / n_total
    n_red = int(round(rate * 100))
    d = Drawing(W, 120)
    s, gap = 9, 2.5
    x0 = 10
    for i in range(100):
        r, c = divmod(i, 20)
        d.add(Rect(x0 + c * (s + gap), 100 - r * (s + gap), s, s, fillColor=RED if i >= 100 - n_red else colors.HexColor("#c3c2b7"),
                   strokeColor=None, rx=1.5, ry=1.5))
    d.add(String(x0 + 20 * (s + gap) + 18, 92, f"About {n_red} in every 100 companies", fontName="Sans-Bold", fontSize=11, fillColor=INK))
    d.add(String(x0 + 20 * (s + gap) + 18, 76, f"went bankrupt ({rate:.1%} of {n_total:,}).", fontName="Sans", fontSize=10, fillColor=INK2))
    d.add(String(x0 + 20 * (s + gap) + 18, 52, "A lazy model that always says “safe”", fontName="Sans", fontSize=9.5, fillColor=INK2))
    d.add(String(x0 + 20 * (s + gap) + 18, 38, f"would be right {1 - rate:.0%} of the time, yet useless.", fontName="Sans", fontSize=9.5, fillColor=INK2))
    return d


def drawing_split():
    d = Drawing(W, 52)
    w85 = (W - 20) * 0.85
    d.add(Rect(10, 14, w85, 24, fillColor=BLUE, strokeColor=None, rx=3, ry=3))
    d.add(Rect(10 + w85 + 3, 14, (W - 20) * 0.15 - 3, 24, fillColor=AMBER, strokeColor=None, rx=3, ry=3))
    d.add(String(10 + w85 / 2, 22, "85%  study material: training, choosing ratios, tuning, calibration",
                 fontName="Sans-Bold", fontSize=8.5, fillColor=colors.white, textAnchor="middle"))
    d.add(String(10 + w85 + (W - 20) * 0.075, 22, "15% exam", fontName="Sans-Bold", fontSize=8.5,
                 fillColor=INK, textAnchor="middle"))
    d.add(String(10, 2, "Both parts keep the same share of bankrupt companies (a “stratified” split).",
                 fontName="Sans", fontSize=8, fillColor=INK2))
    return d


def drawing_cv():
    d = Drawing(W, 120)
    bw, bh, x0 = 50, 14, 62
    for r in range(5):
        y = 92 - r * 20
        d.add(String(10, y + 3, f"Round {r + 1}", fontName="Sans", fontSize=8.5, fillColor=INK2))
        for c in range(5):
            test = c == r
            d.add(Rect(x0 + c * (bw + 3), y, bw, bh, fillColor=AMBER if test else BLUE_BG,
                       strokeColor=AMBER if test else BLUE, strokeWidth=0.6, rx=2, ry=2))
            d.add(String(x0 + c * (bw + 3) + bw / 2, y + 4, "check" if test else "learn", fontName="Sans",
                         fontSize=7.5, fillColor=INK, textAnchor="middle"))
    d.add(String(x0 + 5 * (bw + 3) + 12, 72, "Each part is used once", fontName="Sans", fontSize=8.5, fillColor=INK2))
    d.add(String(x0 + 5 * (bw + 3) + 12, 60, "for checking, so every", fontName="Sans", fontSize=8.5, fillColor=INK2))
    d.add(String(x0 + 5 * (bw + 3) + 12, 48, "training company gets a", fontName="Sans", fontSize=8.5, fillColor=INK2))
    d.add(String(x0 + 5 * (bw + 3) + 12, 36, "fair, unseen prediction.", fontName="Sans", fontSize=8.5, fillColor=INK2))
    return d


def drawing_tree():
    d = Drawing(W, 175)
    cx = W / 2
    lw, nw = 108, 176                       # leaf and node widths
    leaf_c = [W * 0.125, W * 0.375, W * 0.625, W * 0.875]
    node_c = [W * 0.25, W * 0.75]
    box(d, cx - 95, 140, 190, 30, ["Costs per unit of sales (Attr58)", "more than 1.0?"], size=8)
    box(d, node_c[0] - nw / 2, 80, nw, 30, ["Interest cover (Attr27)", "below 0.5?"], size=8)
    box(d, node_c[1] - nw / 2, 80, nw, 30, ["Quick ratio (Attr46)", "below 0.3?"], size=8)
    leaves = [("+0.9  risk up", RED_BG, RED), ("+0.2", RED_BG, RED), ("+0.1", RED_BG, RED),
              ("−0.4  risk down", BLUE_BG, BLUE)]
    for c, (t, f, e) in zip(leaf_c, leaves):
        box(d, c - lw / 2, 18, lw, 26, [t], fill=f, edge=e, size=8)
    arrow(d, cx - 40, 140, node_c[0], 111)
    arrow(d, cx + 40, 140, node_c[1], 111)
    d.add(String((cx - 40 + node_c[0]) / 2 - 16, 128, "yes", fontName="Sans", fontSize=8, fillColor=INK2))
    d.add(String((cx + 40 + node_c[1]) / 2 + 6, 128, "no", fontName="Sans", fontSize=8, fillColor=INK2))
    for n, (a, b) in zip([0, 0, 1, 1], [(0, "yes"), (1, "no"), (2, "yes"), (3, "no")]):
        x1 = node_c[n] + (-25 if a % 2 == 0 else 25)
        arrow(d, x1, 80, leaf_c[a], 45)
        d.add(String((x1 + leaf_c[a]) / 2 + (6 if leaf_c[a] > x1 else -18), 63, b, fontName="Sans", fontSize=8, fillColor=INK2))
    leaves_range = sorted(H[h]["params"]["num_leaves"] for h in HS)
    d.add(String(cx, 2, f"Simplified illustration. The real trees are learned from data and have "
                        f"{leaves_range[0]}–{leaves_range[-1]} leaves each.",
                 fontName="Sans-Italic", fontSize=7.5, fillColor=MUTED, textAnchor="middle"))
    return d


def drawing_architecture():
    d = Drawing(W, 230)
    lw = W / 2 - 18
    xl, xr = 0, W / 2 + 18
    d.add(Rect(xl, 0, lw, 225, fillColor=PANEL, strokeColor=None, rx=6, ry=6))
    d.add(Rect(xr, 0, lw, 225, fillColor=PANEL, strokeColor=None, rx=6, ry=6))
    d.add(String(xl + lw / 2, 208, "TRAINING  (done once, ~20 minutes)", fontName="Sans-Bold", fontSize=9.5, fillColor=INK, textAnchor="middle"))
    d.add(String(xr + lw / 2, 208, "USING THE APP  (every prediction, < 1 s)", fontName="Sans-Bold", fontSize=9.5, fillColor=INK, textAnchor="middle"))
    bw, bh = lw - 40, 34
    left = [["data/raw/*.arff", "43,405 company statements"],
            ["bankruptcy/train.py", "learn, tune, calibrate, test"],
            ["models/  +  reports/", "5 model files, metadata, charts"]]
    right = [["app.py  (Streamlit)", "you type the ratios"],
             ["bankruptcy/predict.py", "loads models, scores, explains"],
             ["Result on screen", "probability, verdict, reasons"]]
    for col, x in [(left, xl), (right, xr)]:
        for i, lines in enumerate(col):
            y = 150 - i * 62
            box(d, x + 20, y, bw, bh, lines, size=8.5)
            if i < 2:
                arrow(d, x + lw / 2, y, x + lw / 2, y - 26)
    arrow(d, xl + lw - 20, 41, xr + 20, 103, color=BLUE, w=1.5)
    d.add(String(W / 2 + 24, 66, "saved models", fontName="Sans-Italic", fontSize=8, fillColor=BLUE))
    return d


def drawing_risk_scale():
    d = Drawing(W, 62)
    x0, w = 10, W - 20
    med, high = m1["risk_bands"]["medium"], m1["risk_bands"]["high"]

    def X(p):   # stretch the low end so the bands are readable
        return x0 + w * (np.sqrt(p))
    d.add(Rect(X(0), 24, X(med) - X(0), 16, fillColor=GREEN, strokeColor=None))
    d.add(Rect(X(med), 24, X(high) - X(med), 16, fillColor=AMBER, strokeColor=None))
    d.add(Rect(X(high), 24, X(1) - X(high), 16, fillColor=RED, strokeColor=None))
    for p, lab, anchor in [(0, "0%", "start"), (med, f"{med:.1%}", "middle"), (high, f"{high:.0%}", "middle"),
                           (1, "100%", "end")]:
        d.add(String(X(p), 12, lab, fontName="Sans", fontSize=7.5, fillColor=INK2, textAnchor=anchor))
    d.add(String((X(0) + X(med)) / 2, 45, "Low", fontName="Sans-Bold", fontSize=8.5, fillColor=INK, textAnchor="middle"))
    d.add(String((X(med) + X(high)) / 2, 45, "Medium", fontName="Sans-Bold", fontSize=8.5, fillColor=INK, textAnchor="middle"))
    d.add(String((X(high) + X(1)) / 2, 45, "High  →  flagged as likely bankrupt", fontName="Sans-Bold", fontSize=8.5, fillColor=INK, textAnchor="middle"))
    d.add(Line(X(p_bad), 21, X(p_bad), 43, strokeColor=INK, strokeWidth=2))
    d.add(String(X(p_bad) - 5, 1, f"distressed example: {p_bad:.1%}", fontName="Sans-Bold", fontSize=7.5,
                 fillColor=INK, textAnchor="end"))
    return d


def drawing_app():
    d = Drawing(W, 200)
    d.add(Rect(0, 0, W, 196, fillColor=colors.white, strokeColor=LINE, rx=6, ry=6))
    d.add(Rect(0, 0, 130, 196, fillColor=PANEL, strokeColor=None, rx=6, ry=6))
    side = [("Prediction horizon", "1–5 years ahead"), ("This model", "recall, precision, F1, AUC"),
            ("Empty boxes mean…", "unknown / not reported")]
    for i, (a, b) in enumerate(side):
        y = 160 - i * 52
        d.add(String(10, y, a, fontName="Sans-Bold", fontSize=8, fillColor=INK))
        d.add(String(10, y - 12, b, fontName="Sans", fontSize=7.5, fillColor=INK2))
    d.add(String(145, 176, "Bankruptcy Risk Predictor", fontName="Sans-Bold", fontSize=11, fillColor=INK))
    for i, t in enumerate(["Single company", "Batch scoring (CSV)", "How it works"]):
        d.add(Rect(145 + i * 112, 150, 106, 16, fillColor=BLUE_BG if i == 0 else PANEL, strokeColor=None, rx=3, ry=3))
        d.add(String(145 + i * 112 + 53, 155, t, fontName="Sans", fontSize=7.5, fillColor=INK, textAnchor="middle"))
    d.add(String(145, 134, "Start from: Blank form / Healthy / Borderline / Distressed example", fontName="Sans", fontSize=7.5, fillColor=INK2))
    for r in range(3):
        for c in range(2):
            d.add(Rect(145 + c * 160, 104 - r * 22, 150, 15, fillColor=PANEL, strokeColor=None, rx=2, ry=2))
    d.add(String(145, 40, "20 boxes, one per ratio, each with its meaning and typical range", fontName="Sans", fontSize=7.5, fillColor=INK2))
    d.add(Rect(145, 14, 310, 16, fillColor=BLUE, strokeColor=None, rx=3, ry=3))
    d.add(String(300, 19, "Predict bankruptcy risk", fontName="Sans-Bold", fontSize=8, fillColor=colors.white, textAnchor="middle"))
    return d


def confusion(h):
    cm_ = H[h]["metrics"]["test"]["confusion_matrix"]
    rows = [["", P("<b>Model said: bankrupt</b>", "boxc"), P("<b>Model said: safe</b>", "boxc")],
            [P("<b>Really went bankrupt</b>", "box"), P(f"<b>{cm_['tp']}</b><br/>caught ✔", "boxc"), P(f"<b>{cm_['fn']}</b><br/>missed ✘", "boxc")],
            [P("<b>Really survived</b>", "box"), P(f"<b>{cm_['fp']}</b><br/>false alarm ✘", "boxc"), P(f"<b>{cm_['tn']:,}</b><br/>correctly safe ✔", "boxc")]]
    t = Table(rows, colWidths=[4.2 * cm, 4.6 * cm, 4.6 * cm], rowHeights=[1.0 * cm, 1.3 * cm, 1.3 * cm])
    t.setStyle(TableStyle([
        ("GRID", (1, 1), (-1, -1), 1, colors.white), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (1, 1), (1, 1), GREEN_BG), ("BACKGROUND", (2, 2), (2, 2), GREEN_BG),
        ("BACKGROUND", (2, 1), (2, 1), RED_BG), ("BACKGROUND", (1, 2), (1, 2), AMBER_BG),
    ]))
    t.hAlign = "CENTER"
    return t


# ----------------------------------------------------------------------------- the document
def build():
    story = []
    t1 = H[1]["metrics"]["test"]
    cm1 = t1["confusion_matrix"]

    # ---------------- cover
    story += [Spacer(1, 2.2 * cm),
              P("How the Bankruptcy Risk Predictor works", "title"),
              P("A plain-English guide to the data, the model, the scoring and the app", "subtitle"),
              callout([P("<b>The whole project in one sentence</b>", "box"), Spacer(1, 3),
                       P(f"The app takes {len(H[1]['features'])} financial ratios of a company, gives them to a "
                         f"machine-learning model that learned from {n_total:,} real company statements, and returns "
                         "the chance that the company goes bankrupt 1 to 5 years later, together with the reasons.", "box")]),
              Spacer(1, 18),
              tiles([(f"{n_total:,}", "company statements<br/>learned from"),
                     ("64 → 20", "financial ratios<br/>(best 20 kept)"),
                     ("5", "models, one for each<br/>horizon (1–5 years)"),
                     (f"{avg('roc_auc'):.2f}", "average ROC-AUC<br/>on unseen companies")]),
              Spacer(1, 22), P("What's inside", "h2")]
    contents = ["The big picture: how the pieces fit together", "The data: what the model learns from",
                "Training: how the model learns, step by step", "One prediction, worked through with real numbers",
                "Scoring: how we know the model is good", "What makes a company risky?",
                "What changed from the first version", "Using the app", "Limits, and a glossary"]
    story += bullets([f"<b>{i + 1}.</b> {c}" for i, c in enumerate(contents)])
    story += [Spacer(1, 30),
              P(f"Model version {META['model_version']} · trained {META['trained_on']} · "
                f"{META['algorithm']} (LightGBM {META['libraries']['lightgbm']})", "small"),
              PageBreak()]

    # ---------------- 1. big picture
    story += [P("1. The big picture", "h1"),
              P("Think of the project as two halves: a <b>factory</b> and a <b>shop</b>. The factory (training) "
                "runs once. It reads thousands of past company statements, finds the patterns that came before "
                "bankruptcies, and saves what it learned as five small model files. The shop (the app) uses "
                "those files every time someone asks about a company, and answers in a fraction of a second."),
              drawing_architecture(), Spacer(1, 10),
              P("What happens when you press “Predict”", "h2"),
              flow([("You enter the ratios", f"Up to {len(H[1]['features'])} numbers such as “costs per unit of sales”. Unknown boxes can stay empty."),
                    ("The horizon picks the model", "There is a separate model for 1, 2, 3, 4 and 5 years ahead."),
                    ("Hundreds of small decision trees vote", "Each tree looks at a few ratios and adds or subtracts a little risk."),
                    ("The votes add up to a risk score", "A single number: higher means more danger."),
                    ("The score becomes a probability", "A calibrated formula turns it into a chance between 0% and 100%."),
                    ("The probability is compared with a threshold", "At or above the threshold, the company is flagged as likely to go bankrupt."),
                    ("You see the verdict and the reasons", "Which ratios pushed the risk up, and which pulled it down.")]),
              CondPageBreak(8 * cm),
              P("Where everything lives", "h2"),
              table([["Folder / file", "What it is", "In plain words"],
                     ["data/raw/", "1year.arff … 5year.arff", "The raw data: one file per horizon"],
                     ["bankruptcy/train.py", "Training pipeline", "The factory: builds and tests the models"],
                     ["bankruptcy/predict.py", "Prediction engine", "Loads the models, scores a company, explains why"],
                     ["bankruptcy/features.py", "Ratio dictionary", "Name and meaning of each of the 64 ratios"],
                     ["models/", "model_1y.txt … model_5y.txt, metadata.json", "The learned trees, plus thresholds, scores and examples"],
                     ["reports/", "metrics.md, figures/", "Test results and charts"],
                     ["app.py", "Streamlit app", "The shop: the web page you use"],
                     ["tests/", "Automated checks", "Make sure predictions and the app still work"],
                     ["docs/", "This guide, deployment guide", "Documentation"]],
                    [4.8 * cm, 5.4 * cm, W - 10.2 * cm], zebra=True),
              Spacer(1, 14)]

    # ---------------- 2. data
    story += [P("2. The data", "h1"),
              P("The models learn from the <b>Polish Companies Bankruptcy</b> dataset, published by the UCI Machine "
                "Learning Repository (Tomczak, 2016). It was collected from the EMIS database of emerging-market "
                "companies. Bankrupt companies were analysed over 2000–2012, and companies that kept operating over 2007–2013."),
              P("<b>One row = one company's financial statement</b>, turned into 64 financial ratios, plus a label "
                "saying whether the company later went bankrupt. A <b>ratio</b> divides one number by another so that "
                "small and large companies can be compared. For example, <i>net profit ÷ total assets = 0.05</i> means "
                "the company earns 5 cents of profit for every 1 unit of assets it owns."),
              P("The 64 ratios fall into a few families", "h2"),
              table([["Family", "The question it answers", "Example ratio"],
                     ["Profitability", "Is the company making money?", "Net profit / total assets (Attr1)"],
                     ["Debt (leverage)", "How much does it owe compared with what it owns?", "Total liabilities / total assets (Attr2)"],
                     ["Liquidity", "Can it pay the bills that are due soon?", "(Current assets − inventory) / short-term liabilities (Attr46)"],
                     ["Debt service", "Can its profits cover interest payments?", "Operating profit / financial expenses (Attr27)"],
                     ["Efficiency", "How well does it turn assets into sales?", "Sales / total assets (Attr9)"],
                     ["Cost structure", "Do costs eat up the sales?", "Total costs / total sales (Attr58)"],
                     ["Size and growth", "How big is it, and are sales growing?", "Log of total assets (Attr29), sales growth (Attr21)"]],
                    [3.2 * cm, 6.2 * cm, W - 9.4 * cm], zebra=True),
              Spacer(1, 6),
              CondPageBreak(9 * cm),
              P("Five files, five horizons", "h2"),
              P("The data comes in five files. Each one has a different gap between the financial statement and "
                "the bankruptcy outcome. The file names count the <i>year of the statement</i> within a five-year "
                "window, not the years ahead, so the numbers run in reverse:"),
              table([["Horizon", "File", "Companies", "Went bankrupt", "Share"]] +
                    [[yl(h), H[h]["source_file"], f"{H[h]['n_train'] + H[h]['n_test']:,}",
                      f"{H[h]['n_train_bankrupt'] + H[h]['n_test_bankrupt']:,}",
                      f"{(H[h]['n_train_bankrupt'] + H[h]['n_test_bankrupt']) / (H[h]['n_train'] + H[h]['n_test']):.1%}"]
                     for h in HS],
                    [3.4 * cm, 3.2 * cm, 3.2 * cm, 3.4 * cm, W - 13.2 * cm], zebra=True),
              Spacer(1, 6),
              callout(P("<b>Fixed in version 2:</b> the first version labelled <i>1year.arff</i> as “1 year ahead”. "
                        "According to the dataset's documentation it is the opposite: statements from year 1, "
                        "with the bankruptcy status 5 years later.", "box"), bg=AMBER_BG, edge=AMBER),
              Spacer(1, 10),
              KeepTogether([P("Bankruptcy is rare", "h2"), drawing_pictogram(),
                            P("This imbalance is why the guide talks about <i>recall</i>, <i>precision</i> and "
                              "<i>PR-AUC</i> rather than accuracy. Those scores focus on the bankrupt companies, "
                              "which are the ones that matter.")]),
              P("Missing numbers", "h2"),
              P("About 1.5% of the numbers are missing. LightGBM handles gaps natively. Interestingly, a missing value "
                "can itself be a clue: in the 1-year-ahead data, 96% of companies with no sales-growth figure went "
                "bankrupt, probably because failing companies stop reporting. The app therefore asks what an empty "
                "box means. <b>“I don't know the value”</b> (the default) fills in a typical value. "
                "<b>“Not reported in the accounts”</b> lets the model use what it learned about missing data."),
              PageBreak()]

    # ---------------- 3. training
    p1 = H[1]["params"]
    story += [P("3. Training: how the model learns", "h1"),
              P("The same six steps run separately for each of the five horizons. Together they take about "
                "15–20 minutes on a laptop (<font face='Sans-Bold'>python -m bankruptcy.train</font>)."),
              P("Step 1. Lock away a final exam", "h2"),
              P("Before anything else, 15% of the companies are set aside and not touched until the very end. "
                "They act like a final exam the model has never seen, so the test scores are honest. The split uses "
                "the same random seed as version 1, so both versions are graded on exactly the same companies."),
              drawing_split(),
              P("Step 2. Pick the 20 most useful ratios", "h2"),
              P("A quick model is trained on all 64 ratios using only the training companies, and we measure how "
                "much each ratio helped it separate bankrupt from healthy companies (its “gain”). The top 20 are kept. "
                "In tests, 20 ratios were just as accurate as all 64, and they keep the app's form short. "
                f"{len(common)} ratios made it into all five models, which shows they matter at every horizon."),
              P("Step 3. The model: LightGBM, a team of decision trees", "h2"),
              P("A <b>decision tree</b> is a flowchart of yes/no questions about the ratios. At the end of each path "
                "is a small amount of risk to add or subtract. Here is a simplified example:"),
              drawing_tree(), Spacer(1, 8),
              P("One tree on its own is weak. <b>Gradient boosting</b> builds trees one after another, and each new "
                "tree focuses on the mistakes the previous ones still make. The final risk score is the sum of all "
                f"the trees' small answers. For example, the 1-year model uses {p1['num_boost_round']} trees with up to "
                f"{p1['num_leaves']} leaves each. <b>LightGBM</b> is a fast, popular library for this technique and "
                "is among the most accurate methods for table-shaped data like ours."),
              CondPageBreak(7 * cm),
              P("Step 4. Tune the settings with cross-validation", "h2"),
              P("A tree model has settings: how fast it learns, how big each tree may grow, how many companies each "
                "leaf needs, and how strongly to avoid over-fitting. <b>Optuna</b> tries 40 combinations and learns "
                "which areas look promising. Each combination is judged with <b>5-fold cross-validation</b>: the "
                "training companies are cut into 5 parts, and the model learns on 4 and is checked on the 5th, five times over."),
              drawing_cv(),
              P("The combination with the best average <i>PR-AUC</i> (see section 5) wins. To avoid adding useless "
                "trees, training stops as soon as the checking part stops improving (“early stopping”)."),
              P("Step 5. Make the probabilities honest, and set the threshold", "h2"),
              P("Cross-validation gives every training company a prediction from a model that never saw it. These "
                "out-of-fold predictions are used twice:"),
              *bullets(["<b>Calibration.</b> A small formula (Platt scaling) adjusts the raw scores so that "
                        "“30%” really means about 30 in 100 such companies went bankrupt. The chart below checks this.",
                        "<b>Threshold.</b> Every cut-off from 2% to 95% is tried, and the one with the best F1 score "
                        f"(balance of catching bankruptcies and avoiding false alarms) is kept: "
                        + ", ".join(f"{H[h]['threshold']:.0%} ({h} yr)" for h in HS) + "."]),
              img(FIGURES_DIR / "calibration.png", width=8.2 * cm),
              P("Dots close to the grey diagonal mean the predicted probabilities match what really happened.", "small"),
              P("Step 6. Final exam", "h2"),
              P("The final model is trained on all of the 85% training companies with the winning settings. It then "
                "predicts the 15% exam companies exactly once. Those are the scores reported in section 5."),
              PageBreak()]

    # ---------------- 4. worked example
    top6 = exp_bad.head(6)
    stats = m1["feature_stats"]
    story += [P("4. One prediction, worked through", "h1"),
              P(f"Let's follow a real company from the 1-year model's exam set (the app's “Distressed company” "
                f"example). The model had never seen it, and the company really did go bankrupt. Here are the six "
                f"ratios that influenced its result most:"),
              table([["Ratio", "This company", "Typical company", "Push on risk"]] +
                    [[f"{r.meaning} ({r.feature})", f"{r.value:,.3f}", f"{stats[r.feature]['median']:,.3f}",
                      f"{r.contribution:+.2f}"] for r in top6.itertuples()],
                    [8.2 * cm, 2.9 * cm, 3.0 * cm, W - 14.1 * cm], zebra=True),
              Spacer(1, 6),
              img(chart_example(), width=W * 0.92),
              P("Red bars push towards bankruptcy, blue bars push away from it. These pushes are SHAP values, "
                "computed exactly by LightGBM for every prediction.", "small"),
              P("From pushes to a probability", "h2"),
              *bullets([f"<b>Start</b> from the score of an average company: <b>{base_bad:+.2f}</b>.",
                        f"<b>Add</b> the pushes of all {len(m1['features'])} ratios: "
                        f"<b>{exp_bad['contribution'].sum():+.2f}</b>. Risk score = <b>{z_bad:+.2f}</b>.",
                        "<b>Squash</b> the score into a probability with the formula 1 ÷ (1 + e<super>−score</super>). "
                        "Any score becomes a number between 0% and 100%: big positive scores land near 100%, big "
                        f"negative ones near 0%. Here: <b>{p_bad:.1%}</b>.",
                        f"<b>Compare</b> with the 1-year threshold of {m1['threshold']:.0%}: {p_bad:.1%} is above it, "
                        "so the verdict is <b>likely to go bankrupt</b> and the risk level is <b>High</b>."]),
              img(chart_sigmoid(), width=W * 0.78),
              P(f"For contrast, the “Healthy company” example scores {z_good:+.2f}, which becomes a probability of "
                f"{p_good:.1%}. That is far below the threshold, so it is rated Low risk.", "small"),
              P("Risk levels", "h2"),
              P("Besides the yes/no verdict, the app shows a risk level. <b>High</b> means at or above the threshold. "
                "<b>Medium</b> means above half the threshold: not flagged, but worth watching. <b>Low</b> is everything "
                "below. For the 1-year model:"),
              drawing_risk_scale(),
              PageBreak()]

    # ---------------- 5. scoring
    rate1 = (H[1]["n_test_bankrupt"]) / H[1]["n_test"]
    story += [P("5. Scoring: how we know it is good", "h1"),
              P(f"All scores come from the locked-away exam companies. Take the 1-year model: its exam had "
                f"{H[1]['n_test']:,} companies, {H[1]['n_test_bankrupt']} of which really went bankrupt. "
                "Every exam company ends up in one of four boxes:"),
              confusion(1), Spacer(1, 10),
              table([["Score", "What it means in plain words", "1-year model"],
                     ["Recall", f"Of the companies that really went bankrupt, how many did we catch? "
                                f"({cm1['tp']} of {cm1['tp'] + cm1['fn']})", f"{t1['recall']:.0%}"],
                     ["Precision", f"Of the companies we flagged, how many really went bankrupt? "
                                   f"({cm1['tp']} of {cm1['tp'] + cm1['fp']})", f"{t1['precision']:.0%}"],
                     ["F1 score", "One number that balances recall and precision (0 = useless, 1 = perfect)", f"{t1['f1']:.2f}"],
                     ["ROC-AUC", "Pick one bankrupt and one healthy company at random: how often does the bankrupt one get the higher risk score? 0.5 = coin flip, 1 = always.", f"{t1['roc_auc']:.2f}"],
                     ["PR-AUC", f"Average precision across all thresholds. A random guess would score about {rate1:.2f} (the bankruptcy rate); 1 = perfect.", f"{t1['pr_auc']:.2f}"],
                     ["Accuracy", f"Share of all companies classified correctly. Looks great, but a model that always says “safe” already gets {1 - rate1:.0%}.", f"{t1['accuracy']:.0%}"]],
                    [2.4 * cm, W - 4.8 * cm, 2.4 * cm], zebra=True),
              Spacer(1, 10),
              P("Results for all five horizons", "h2"),
              table([["Horizon", "Threshold", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"]] +
                    [[yl(h), f"{H[h]['threshold']:.0%}", f"{H[h]['metrics']['test']['recall']:.0%}",
                      f"{H[h]['metrics']['test']['precision']:.0%}", f"{H[h]['metrics']['test']['f1']:.2f}",
                      f"{H[h]['metrics']['test']['roc_auc']:.2f}", f"{H[h]['metrics']['test']['pr_auc']:.2f}"] for h in HS] +
                    [["Average", "", f"{avg('recall'):.0%}", f"{avg('precision'):.0%}", f"{avg('f1'):.2f}",
                      f"{avg('roc_auc'):.2f}", f"{avg('pr_auc'):.2f}"]],
                    [3.2 * cm] + [(W - 3.2 * cm) / 6] * 6, zebra=True),
              Spacer(1, 6),
              P("Each exam has only 40–80 bankrupt companies, so the scores carry some uncertainty. For example, the "
                f"1-year F1 of {t1['f1']:.2f} could plausibly be anywhere from {H[1]['metrics']['test_ci95']['f1'][0]:.2f} "
                f"to {H[1]['metrics']['test_ci95']['f1'][1]:.2f} (95% bootstrap range). Differences of a few "
                "points between horizons or versions should not be over-read. The ranges for every horizon are in "
                "reports/metrics.md."),
              PageBreak()]

    # ---------------- 6. what makes a company risky
    dirs = directions(1, list(H[1]["importance"])[:6])
    story += [P("6. What makes a company risky?", "h1"),
              P("Averaging the SHAP pushes over all training companies shows which ratios each model relies on "
                "most. For the 1-year model:"),
              img(FIGURES_DIR / "importance_1y.png", width=W * 0.84),
              Spacer(1, 4),
              table([["Ratio", "What it measures", "What the data shows"]] +
                    [[f"{f}", meaning(f), dirs[f]] for f in list(H[1]["importance"])[:6]],
                    [1.7 * cm, 5.6 * cm, W - 7.3 * cm], zebra=True),
              Spacer(1, 6),
              P("Why a bar can look surprising: many ratios measure similar things (several of them are "
                "profit ratios, for example). The model may give the credit for a warning sign to one ratio and "
                "use a related one to fine-tune the result, so a single bar can sometimes point the “wrong” way. "
                "Read a company's reasons as a group rather than one bar at a time.", "small"),
              Spacer(1, 4),
              P(f"<b>{len(common)} ratios appear in all five models</b> ({', '.join(common)}). They cover the "
                "classic warning signs analysts look for: weak interest cover, costs close to or above sales, thin "
                "liquidity, poor cash flow relative to debt, little accumulated profit and shrinking sales. The "
                "model's job is to weigh them against each other far more precisely than a rule of thumb can."),
              P("Charts for the other horizons are in reports/figures/ (importance_2y.png … importance_5y.png) "
                "and in the app's “How it works” tab.", "small"),
              PageBreak()]

    # ---------------- 7. v1 -> v2
    story += [P("7. What changed from the first version", "h1"),
              P("Version 1 (the original notebook) was a good start. Version 2 keeps the same idea (LightGBM, one "
                "model per horizon, a tuned threshold) and fixes the issues below."),
              table([["Problem in version 1", "Fix in version 2", "Why it matters"],
                     ["Horizons were reversed", "Mapped as the dataset defines them", "“1 year ahead” now really means 1 year ahead"],
                     ["The top ratios were chosen using all companies, exam included", "Chosen from training companies only", "Exam scores are honest (no “peeking”)"],
                     ["Settings set by hand; one of them (subsample) silently did nothing", "40 settings searched with Optuna and 5-fold CV", "Better, more reliable models"],
                     ["Threshold picked on one small validation split", "Picked on out-of-fold predictions for all training companies", "Threshold is stable, not luck"],
                     ["Probabilities distorted by class weighting", "Calibrated with Platt scaling", "“30%” means 30%"],
                     ["The app ignored the tuned thresholds", "App uses each horizon's threshold", "Verdicts match the tested model"],
                     ["No explanation of results", "SHAP reasons for every prediction", "You can see why"],
                     ["82 MB of version-locked pickle files", "5 MB of plain LightGBM text files", "Easy to deploy, safe to load"],
                     ["One notebook", "Package, training command, 11 tests, Docker, deployment guide", "Reproducible and maintainable"]],
                    [5.2 * cm, 5.6 * cm, W - 10.8 * cm], zebra=True),
              Spacer(1, 4),
              P("Scores on the same exam companies", "h2"),
              img(FIGURES_DIR / "old_vs_new.png", width=W * 0.8),
              table([["Average over 5 horizons", "Version 1", "Version 2"],
                     ["PR-AUC", f"{np.mean([LEGACY_V1[h]['pr_auc'] for h in HS]):.3f}", f"{avg('pr_auc'):.3f}  (better on all 5)"],
                     ["ROC-AUC", f"{np.mean([LEGACY_V1[h]['roc_auc'] for h in HS]):.3f}", f"{avg('roc_auc'):.3f}  (better or equal on all 5)"],
                     ["F1", f"{np.mean([LEGACY_V1[h]['f1'] for h in HS]):.3f}", f"{avg('f1'):.3f}"]],
                    [6 * cm, 4 * cm, W - 10 * cm]),
              Spacer(1, 6),
              P("F1 depends on one threshold and a small exam, so it moves around more: version 2 is higher on "
                "average but slightly lower on three horizons. PR-AUC and ROC-AUC look at all thresholds at once "
                "and improved across the board. Version 1 also had a small unfair advantage: its ratios were "
                "chosen with the exam companies in view.", "small"),
              PageBreak()]

    # ---------------- 8. app
    story += [P("8. Using the app", "h1"),
              P("Start it with <font face='Sans-Bold'>streamlit run app.py</font>, or open the hosted link once it "
                "is deployed (see docs/DEPLOYMENT.md). The screen has a sidebar and three tabs:"),
              drawing_app(), Spacer(1, 10),
              table([["Part", "What you do there"],
                     ["Sidebar: horizon", "Choose how many years ahead to predict. The form and the model change with it."],
                     ["Sidebar: this model", "See how well the chosen model did on its exam: recall, precision, F1 and ROC-AUC."],
                     ["Sidebar: empty boxes", "“I don't know the value” fills gaps with typical values (recommended). "
                                              "“Not reported in the accounts” treats them as genuinely missing, which the model reads as a warning sign."],
                     ["Single company tab", "Type the ratios, or load a real example company (healthy, borderline or "
                                            "distressed), then press Predict. You get the probability, threshold, risk level, verdict, and a chart of the reasons."],
                     ["Batch scoring tab", "Download the CSV template, fill one row per company, upload it, and get "
                                           "probabilities and verdicts for all five horizons in one table you can download."],
                     ["How it works tab", "A short version of this guide, the test results and the charts."]],
                    [4.2 * cm, W - 4.2 * cm], zebra=True),
              Spacer(1, 10),
              callout(P("<b>Tip:</b> hover over the small question-mark icon next to each box to see the ratio's exact formula and the "
                        "typical range in the data. Values far outside that range are unusual and worth double-checking.", "box")),
              PageBreak()]

    # ---------------- 9. limits + glossary
    story += [P("9. Limits, and a glossary", "h1"),
              P("Use it wisely", "h2"),
              *bullets([
                  "<b>Where the data comes from.</b> The models learned from Polish companies around 2000–2013. "
                  "Companies in other countries, industries or economic periods may behave differently, so check "
                  "results on your own data before relying on them.",
                  "<b>Different collection periods.</b> Bankrupt and surviving companies were collected over different "
                  "years. Part of what a model picks up may be differences between those periods rather than "
                  "financial distress. This may be why some horizons score surprisingly well.",
                  "<b>Same companies in several files.</b> A company can appear in more than one year file. That is "
                  "why each horizon is modelled and tested separately. Mixing the files made scores look far "
                  "better than they really were.",
                  "<b>Small exams.</b> 40–80 bankrupt companies per horizon means a few points of uncertainty in every score.",
                  "<b>A screening tool, not a verdict.</b> Use it to decide which companies deserve a closer look, "
                  "and combine it with human judgement. It is not financial advice."]),
              P("Glossary", "h2"),
              table([["Term", "Meaning"],
                     ["Ratio / feature", "One input number, e.g. total costs ÷ total sales."],
                     ["Horizon", "How many years ahead the model predicts (1–5)."],
                     ["Decision tree", "A flowchart of yes/no questions that ends in a small amount of risk."],
                     ["Gradient boosting / LightGBM", "Many small trees built in sequence, each fixing earlier mistakes. LightGBM is the library used."],
                     ["Training / test set", "Companies the model learns from (85%) vs. the locked-away final exam (15%)."],
                     ["Cross-validation", "Learning on 4 parts of the training data and checking on the 5th, five times, to judge settings fairly."],
                     ["Hyper-parameters / tuning", "The model's settings, and the search for the best ones (done with Optuna)."],
                     ["Early stopping", "Stop adding trees once they no longer improve results on unseen data."],
                     ["Risk score (log-odds)", "The model's raw output. Higher means riskier, with no fixed range."],
                     ["Calibration", "Adjusting scores so the probabilities match real-world frequencies."],
                     ["Threshold", "The probability at or above which a company is flagged as likely bankrupt."],
                     ["SHAP value", "How much one ratio pushed one company's score up or down."],
                     ["Recall / precision / F1", "Share of bankruptcies caught / share of alarms that were right / their balance."],
                     ["ROC-AUC / PR-AUC", "Threshold-free scores of how well risky companies are ranked above safe ones."],
                     ["Data leakage", "Accidentally letting exam information into training, which makes scores look too good."]],
                    [4.6 * cm, W - 4.6 * cm], zebra=True)]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="How the Bankruptcy Risk Predictor works",
                            author="Subhojit Sapui, Ayush Ranjan, Aditi Pandey",
                            subject="Plain-English guide to the bankruptcy prediction project")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
