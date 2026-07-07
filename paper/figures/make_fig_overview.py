"""Figure 3: ReQueST protocol and scoring overview (v3).

Design refined through GPT-5 + Gemini critique rounds: real frames on a
timeline, then three response rows whose evaluation is SHOWN, not told.
The grouped statement's spans carry highlighter tints that match the
per-reading grid cells (span-to-cell color matching, no arrows), Clarify
shows its two variants with credits in the pills, Commit shows an all-gray
grid. Style copied from gpt-image references (scratchpad design_v1/v2.png).
Output: paper/figures/fig_overview.{pdf,png}, full text width.
"""
import pathlib

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
from PIL import Image

OUT = pathlib.Path(__file__).resolve().parent

BG = "#f6f4ee"
CARD = "#fdfcf9"
CARD_EC = "#e3ded2"
GREEN_D = "#2e7f60"
BLUE_D = "#3c6ca8"
RED_D = "#c5544a"
AMBER_D = "#a8742a"
T_GREEN = "#cfe7db"   # highlighter / cell tints
T_AMBER = "#f2e0b4"
T_BLUE = "#d3e0f0"
T_GRAY = "#e9e6df"
R_GREEN = "#eef5f1"   # row-card tints
R_BLUE = "#eef2f8"
R_RED = "#faf0ee"
INK = "#2f2f2f"
GRAY = "#8f8b83"
TEAL = "#3f7f72"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FW, FH = 6.8, 4.0
U = 100 * FH / FW  # 58.8
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, U)
ax.axis("off")
ax.add_patch(Rectangle((0, 0), 100, U, ec="none", fc=BG, zorder=0))


def rbox(x, y, w, h, fc, ec="none", lw=0.8, r=1.2, z=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       ec=ec, fc=fc, lw=lw, zorder=z)
    ax.add_patch(b)
    return b


# ---------------- video frames on a timeline ----------------
teaser = Image.open(OUT / "figure1.png")
CROPS = [(20, 60, 664, 352), (678, 60, 1322, 352), (1336, 60, 1998, 352)]
frames = [teaser.crop(c) for c in CROPS]

FRW = 27.0
FRH = FRW * 292 / 644
FY1 = U - 1.2
FY0 = FY1 - FRH
FXS = [2.0, 36.5, 71.0]
for fx, fr in zip(FXS, frames):
    im = ax.imshow(fr, extent=(fx, fx + FRW, FY0, FY1), aspect="auto", zorder=3)
    clip = FancyBboxPatch((fx, FY0), FRW, FRH,
                          boxstyle="round,pad=0,rounding_size=0.9",
                          transform=ax.transData)
    im.set_clip_path(clip)
for gx in [(FXS[0] + FRW + FXS[1]) / 2, (FXS[1] + FRW + FXS[2]) / 2]:
    ax.text(gx, (FY0 + FY1) / 2, r"$\cdots$", fontsize=13, color=GRAY,
            ha="center", va="center", zorder=4)

TLY = FY0 - 1.7
ax.add_patch(FancyArrowPatch((2.0, TLY), (98.5, TLY), arrowstyle="-|>",
                             mutation_scale=9, lw=1.1, color=TEAL,
                             shrinkA=0, shrinkB=0, zorder=3))
for fx, (ts, tag) in zip(FXS, [("0:12", "attempt 1"), ("0:58", "attempt 2"),
                               ("3:26", "attempt 3")]):
    cx = fx + FRW / 2
    ax.plot([cx, cx], [TLY - 0.55, TLY + 0.55], color=TEAL, lw=1.1, zorder=4)
    ax.text(cx, TLY - 1.7, f"{ts}  {tag}", fontsize=6.6, color=INK,
            ha="center", va="center", zorder=4)
ax.text(98.5, TLY - 1.7, "time", fontsize=6.2, color=GRAY, ha="right",
        va="center", zorder=4)

# ---------------- zone headers ----------------
HDY = TLY - 4.4
ax.text(12.5, HDY, "Question, $K$ = 3 readings", fontsize=7.4,
        fontweight="bold", color=INK, ha="center", va="center", zorder=4)
ax.text(44.0, HDY, "Model's first response", fontsize=7.4,
        fontweight="bold", color=INK, ha="center", va="center", zorder=4)
ax.text(81.5, HDY, "Per-reading credit", fontsize=7.4,
        fontweight="bold", color=INK, ha="center", va="center", zorder=4)

SY1 = HDY - 1.8   # top of the cards zone
SY0 = 3.0

# ---------------- left question card ----------------
LX, LW = 1.5, 22.0
rbox(LX, SY0, LW, SY1 - SY0, fc=CARD, ec=CARD_EC, lw=0.9, r=1.6, z=1)
bx0, bw0 = LX + 1.4, LW - 2.8
# person avatar + bubble
chip_c = (bx0 + 1.6, SY1 - 3.2)
ax.add_patch(Circle(chip_c, 1.5, ec="none", fc="#dcebe3", zorder=4))
ax.add_patch(Circle((chip_c[0], chip_c[1] + 0.40), 0.50, ec="none", fc=GREEN_D, zorder=5))
ax.add_patch(Ellipse((chip_c[0], chip_c[1] - 0.60), 1.65, 1.1, ec="none",
                     fc=GREEN_D, zorder=5))
rbox(bx0 + 3.6, SY1 - 8.6, bw0 - 3.6, 7.2, fc="white", ec=CARD_EC, lw=0.8, r=1.0)
ax.text(bx0 + 3.6 + (bw0 - 3.6) / 2, SY1 - 5.0, "Does the ball go\ninto the door?",
        fontsize=6.9, color=INK, ha="center", va="center", style="italic",
        zorder=4, linespacing=1.3)
ax.text(bx0 + 0.3, SY1 - 12.4, "gold readings and answers", fontsize=6.2,
        color=GRAY, ha="left", va="center", zorder=4)
for i, (label, ans, ac) in enumerate([("1  first attempt", "no", AMBER_D),
                                      ("2  second attempt", "yes", GREEN_D),
                                      ("3  third attempt", "yes", GREEN_D)]):
    yy = SY1 - 15.8 - i * 5.6
    tl = ax.text(bx0 + 0.3, yy, label, fontsize=6.5, color=INK, ha="left",
                 va="center", zorder=4)
    ax.text(bx0 + bw0 - 0.3, yy, ans, fontsize=6.5, color=ac,
            fontweight="bold", ha="right", va="center", zorder=4)
    if i == 1:
        ax.annotate("intended", xycoords=tl, xy=(1, 0.5),
                    textcoords="offset points", xytext=(3.5, 0), fontsize=5.0,
                    color=BLUE_D, ha="left", va="center", zorder=5,
                    style="italic",
                    bbox=dict(boxstyle="round,pad=0.18", fc=T_BLUE, ec="none"))
    if i < 2:
        ax.plot([bx0 + 0.2, bx0 + bw0 - 0.2], [yy - 2.8, yy - 2.8],
                color=CARD_EC, lw=0.6, zorder=3)

# ---------------- response rows ----------------
RXZ, RWZ = 25.5, 73.0          # rows zone
BUBX, BUBW = RXZ + 1.6, 36.0   # utterance bubbles
GRIDX = 64.5                   # reading-cell grid
CELLW, CELLH, CGAP = 7.4, 3.4, 0.8
BDGX = 94.0                    # credit badge center


def cellrow(ycen, fills, texts, tcolors):
    for j in range(3):
        x = GRIDX + j * (CELLW + CGAP)
        rbox(x, ycen - CELLH / 2, CELLW, CELLH, fc=fills[j], r=0.7, z=3)
        ax.text(x + CELLW / 2, ycen, texts[j], fontsize=6.4, color=tcolors[j],
                ha="center", va="center", zorder=4, fontweight="bold")


def badge(ycen, text, color):
    w = 7.5
    rbox(BDGX - w / 2, ycen - 1.8, w, 3.6, fc=color, r=1.0, z=3)
    ax.text(BDGX, ycen, text, fontsize=6.8, color="white", fontweight="bold",
            ha="center", va="center", zorder=4)


ROW1H, ROW2H, ROW3H, RGAP = 9.4, 15.2, 7.0, 1.3
R1Y = SY1 - ROW1H
R2Y = R1Y - RGAP - ROW2H
R3Y = R2Y - RGAP - ROW3H

# --- row 1: Enumerate ---
rbox(RXZ, R1Y, RWZ, ROW1H, fc=R_GREEN, ec=CARD_EC, lw=0.8, r=1.4, z=1)
ax.text(RXZ + 1.6, R1Y + ROW1H - 1.9, "Enumerate", fontsize=7.2,
        color=GREEN_D, fontweight="bold", ha="left", va="center", zorder=4)
ax.text(RXZ + 11.2, R1Y + ROW1H - 1.9, "(singly or grouped)", fontsize=6.2,
        color=GRAY, ha="left", va="center", zorder=4)
ax.text(GRIDX + 1.5 * CELLW + CGAP, R1Y + ROW1H - 1.9,
        "LLM judge assigns each reading", fontsize=5.9, color=GRAY,
        ha="center", va="center", zorder=4)
yq = R1Y + 2.9
rbox(BUBX, yq - 2.2, BUBW, 4.4, fc="white", ec=CARD_EC, lw=0.8, r=1.0, z=2)
t0 = ax.text(BUBX + 1.8, yq, "“", fontsize=6.9, color=INK, ha="left",
             va="center", zorder=5)
t = ax.annotate("Every attempt succeeds", xycoords=t0, xy=(1, 0.5),
                textcoords="offset points", xytext=(1.0, 0), fontsize=6.9,
                color=INK, ha="left", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.16", fc=T_GREEN, ec="none"))
ax.annotate("2,3", xycoords=t, xy=(1, 1), textcoords="offset points",
            xytext=(0.5, 0.6), fontsize=5.6, color=GREEN_D, ha="left",
            va="bottom", zorder=5, fontweight="bold")
t2 = ax.annotate("except the first.", xycoords=t, xy=(1, 0.5),
                 textcoords="offset points", xytext=(8.5, 0), fontsize=6.9,
                 color=INK, ha="left", va="center", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.16", fc=T_AMBER, ec="none"))
ax.annotate("1", xycoords=t2, xy=(1, 1), textcoords="offset points",
            xytext=(0.5, 0.6), fontsize=5.6, color=AMBER_D, ha="left",
            va="bottom", zorder=5, fontweight="bold")
ax.annotate("”", xycoords=t2, xy=(1, 0.5), textcoords="offset points",
            xytext=(1.2, 0), fontsize=6.9, color=INK, ha="left", va="center",
            zorder=5)
cellrow(yq, [T_AMBER, T_GREEN, T_GREEN], ["1  no", "2  yes", "3  yes"],
        [AMBER_D, GREEN_D, GREEN_D])
badge(yq, "3/3", GREEN_D)

# --- row 2: Clarify ---
rbox(RXZ, R2Y, RWZ, ROW2H, fc=R_BLUE, ec=CARD_EC, lw=0.8, r=1.4, z=1)
ax.text(RXZ + 1.6, R2Y + ROW2H - 1.9, "Clarify", fontsize=7.2, color=BLUE_D,
        fontweight="bold", ha="left", va="center", zorder=4)
ya = R2Y + ROW2H - 5.2
yb = ya - 4.2
yf = R2Y + 1.8
for (yy, tag, quote) in [(ya, "scope-anchored", "“Which attempt do you mean?”"),
                         (yb, "vague", "“Could you be more specific?”")]:
    rbox(BUBX, yy - 1.7, BUBW, 3.4, fc="white", ec=CARD_EC, lw=0.8, r=1.0, z=2)
    ax.text(BUBX + 10.0, yy, tag, fontsize=6.0, color=BLUE_D, ha="right",
            va="center", zorder=5, style="italic")
    ax.text(BUBX + 11.2, yy, quote, fontsize=6.9, color=INK, ha="left",
            va="center", zorder=5)
badge(ya, "1.0", BLUE_D)
badge(yb, "0.5", BLUE_D)
# fork: both variants lead to the same follow-up
fkx = BUBX - 0.9
ax.plot([fkx, fkx], [ya, yf], color=BLUE_D, lw=0.8, zorder=3)
for yy in (ya, yb):
    ax.plot([fkx, BUBX - 0.1], [yy, yy], color=BLUE_D, lw=0.8, zorder=3)
ax.add_patch(FancyArrowPatch((fkx, yf), (BUBX + 0.9, yf), arrowstyle="-|>",
                             mutation_scale=6, lw=0.8, color=BLUE_D,
                             shrinkA=0, shrinkB=0, zorder=3))
ax.text(BUBX + 1.8, yf, "interlocutor:", fontsize=6.0, color=GRAY, ha="left",
        va="center", zorder=4)
ax.text(BUBX + 9.0, yf, "“The second attempt.”", fontsize=6.9,
        color=INK, ha="left", va="center", zorder=4)
ax.text(BUBX + 24.8, yf, "model:", fontsize=6.0, color=GRAY, ha="left",
        va="center", zorder=4)
ax.text(BUBX + 29.4, yf, "“Yes.”", fontsize=6.9, color=INK,
        ha="left", va="center", zorder=4)
cellrow((ya + yb) / 2, [T_GRAY, T_BLUE, T_GRAY], ["1  –", "2  yes", "3  –"],
        [GRAY, BLUE_D, GRAY])
ax.text(GRIDX + 1.5 * CELLW + CGAP, (ya + yb) / 2 - 3.6,
        "only the intended reading is answered", fontsize=6.0, color=GRAY,
        ha="center", va="center", zorder=4)

# --- row 3: Commit silently ---
rbox(RXZ, R3Y, RWZ, ROW3H, fc=R_RED, ec=CARD_EC, lw=0.8, r=1.4, z=1)
ax.text(RXZ + 1.6, R3Y + ROW3H - 1.9, "Commit silently", fontsize=7.2,
        color=RED_D, fontweight="bold", ha="left", va="center", zorder=4)
yc = R3Y + 2.4
rbox(BUBX, yc - 1.9, 12.0, 3.8, fc="white", ec=CARD_EC, lw=0.8, r=1.0, z=2)
ax.text(BUBX + 6.0, yc, "“Yes.”", fontsize=6.9, color=INK,
        ha="center", va="center", zorder=5)
cellrow(yc, [T_GRAY, T_GRAY, T_GRAY], ["1  ?", "2  ?", "3  ?"],
        [GRAY, GRAY, GRAY])
badge(yc, "0", RED_D)

# ---------------- footer ----------------
ax.text(1.5, 1.4, "ReQueST score = mean credit per question", fontsize=6.8, color=INK,
        fontweight="bold", ha="left", va="center", zorder=4)
ax.text(98.5, 1.4, "credit badges show the per-response score", fontsize=6.2,
        color=GRAY, ha="right", va="center", zorder=4)

fig.savefig(OUT / "fig_overview.pdf")
fig.savefig(OUT / "fig_overview.png", dpi=300)
print("wrote", OUT / "fig_overview.pdf")
