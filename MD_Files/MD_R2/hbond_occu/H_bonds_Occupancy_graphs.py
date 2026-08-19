#!/usr/bin/env python3
# ============================================================
#  HYDROGEN-BOND OCCUPANCY PLOTS
# ------------------------------------------------------------
#  Produces:
#    1) one figure per SYSTEM x ROUND      -> HB_individual_graphs/
#    2) one combined figure                -> rows = systems
#                                             col 1 = Round1
#                                             col 2 = Round2
#
#  Fixes vs. previous version:
#    * legends are parsed by SERIES INDEX (@ sN legend "...") instead of
#      being appended blindly, so label i always belongs to column i
#    * GROMACS formatting escapes (\S \N \s \f{...}) are stripped
#    * duplicate legend names are made unique instead of being silently
#      collapsed by the de-duplication step (that was hiding entries)
#    * every series in a plot gets a UNIQUE colour+linestyle combination
#      (20 colours x 4 linestyles = 80 unique styles) and the same
#      interaction keeps the same style in every plot
#    * the threshold line now carries its own legend entry
#    * a diagnostic report tells you exactly which files had missing
#      or mismatched legend entries
# ============================================================

import os
import re
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

# Set Times New Roman globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = "HB_results"

SYSTEMS = [
    "VOR",
    "AFR1",
    "AFR2",
    "AFR3",
    "AFR4",
    "AFR5",
]

ROUNDS = ["Round1", "Round2"]

# Suffix used to build the panel titles: "AFR1" + "_" + "R1" -> "AFR1_R1"
ROUND_SUFFIX = {
    "Round1": "R1",
    "Round2": "R2",
}


def panel_title(system, round_name):
    return f"{system}_{ROUND_SUFFIX.get(round_name, round_name)}"

XVG_NAME = "plot.xvg"

OUTPUT_DIR = "HB_individual_graphs"
COMBINED_FILE = "HB_combined_Round1_vs_Round2.png"

THRESHOLD = 20.0
THRESHOLD_LABEL = f"{THRESHOLD:g}% occupancy threshold"

DPI = 600            # individual figures
COMBINED_DPI = 300   # combined figure (6 rows x 2 cols -> keep this lower)

Y_LIM = (0, 100)

# X ticks: set a number (e.g. 2000) to force a fixed step,
# or leave None to derive nice ticks from the data itself.
X_TICK_STEP = None

# Optional decluttering: set to a number (e.g. 20) to plot only the
# interactions whose occupancy reaches that value at least once.
# Set to None to plot every series.
HIDE_SERIES_BELOW = None

# ---- font sizes: individual figures ----
FS_IND = {
    "title": 36,
    "label": 44,
    "tick": 30,
    "legend": 18,
}

# ---- font sizes: combined figure ----
FS_COMB = {
    "title": 20,
    "label": 24,
    "tick": 20,
    "legend": 20,
}

MAX_LEGEND_ROWS = 20   # more entries than this -> extra legend columns


# ============================================================
# FONT (falls back gracefully if Times New Roman is not installed)
# ============================================================

def resolve_font(candidates=("Times New Roman", "Liberation Serif",
                             "Nimbus Roman", "DejaVu Serif")):
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return "serif"


FONT = resolve_font()

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [FONT, "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.linewidth"] = 1.5


# ============================================================
# STYLE REGISTRY
#   colour + linestyle are keyed on the interaction name, and assigned
#   globally in order of first appearance, so an interaction looks the
#   same in Round1, Round2 and in the combined figure.
#   20 colours x 4 linestyles = 80 visually distinct combinations.
# ============================================================

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#00429d", "#93003a", "#00b159", "#6a3d9a", "#b15928",
]

LINESTYLES = [
    "-",
    (0, (6, 2)),      # dashed
    (0, (6, 2, 1, 2)),  # dash-dot
    (0, (1, 1.6)),    # dotted
]

interaction_styles = {}   # name -> dict(color=..., linestyle=...)


def get_style(interaction):
    if interaction not in interaction_styles:
        idx = len(interaction_styles)
        interaction_styles[interaction] = {
            "color": PALETTE[idx % len(PALETTE)],
            "linestyle": LINESTYLES[(idx // len(PALETTE)) % len(LINESTYLES)],
        }
    return interaction_styles[interaction]


# ============================================================
# XVG PARSING
# ============================================================

# "@ s0 legend "ARG78-Side""  /  "@s0 legend "...""
SERIES_LEGEND_RE = re.compile(
    r'^@\s*s\s*(\d+)\s+legend\s+"(.*)"\s*$', re.IGNORECASE
)

# older grace syntax: "@ legend string 0 "...""
LEGEND_STRING_RE = re.compile(
    r'^@\s*legend\s+string\s+(\d+)\s+"(.*)"\s*$', re.IGNORECASE
)

# GROMACS / grace formatting escapes: \S \N \s \R \f{Symbol} \x ...
GMX_ESCAPE_RE = re.compile(r'\\(?:f\{[^}]*\}|[sSNRrxfF0-9])')


def clean_name(name):
    name = GMX_ESCAPE_RE.sub("", name)
    name = name.replace('"', "").replace("\\", "")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def make_unique(labels):
    """Two H-bonds can legitimately carry the same descriptive name.
    Keeping them identical made the legend hide one of them, so the
    duplicates get a numeric suffix."""
    seen = {}
    out = []
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            out.append(f"{lab} #{seen[lab]}")
        else:
            seen[lab] = 1
            out.append(lab)
    return out


def parse_xvg(path):
    """Returns dict(x, y, labels, n_series, n_legends, blocks) or None."""
    legend_map = {}
    rows = []
    blocks = 0

    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()

            if not line:
                continue

            if line.startswith("@"):
                m = SERIES_LEGEND_RE.match(line)
                if m:
                    legend_map[int(m.group(1))] = clean_name(m.group(2))
                    continue
                m = LEGEND_STRING_RE.match(line)
                if m:
                    legend_map[int(m.group(1))] = clean_name(m.group(2))
                continue

            if line.startswith("#"):
                continue

            if line.startswith("&"):     # grace dataset separator
                blocks += 1
                continue

            parts = line.replace(",", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue

    if not rows:
        return None

    # tolerate ragged rows by padding with NaN
    width = max(len(r) for r in rows)
    arr = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r

    x = arr[:, 0]
    y = arr[:, 1:]

    labels = [
        legend_map.get(i, f"Series {i + 1} (no legend in file)")
        for i in range(y.shape[1])
    ]

    return {
        "x": x,
        "y": y,
        "labels": make_unique(labels),
        "n_series": y.shape[1],
        "n_legends": len(legend_map),
        "blocks": blocks,
    }


# ============================================================
# PASS 1 - READ EVERYTHING, REGISTER STYLES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA = {}
report = []

for system in SYSTEMS:
    for round_name in ROUNDS:

        xvg_file = os.path.join(BASE_DIR, system, round_name, XVG_NAME)
        key = (system, round_name)

        if not os.path.isfile(xvg_file):
            DATA[key] = None
            report.append((system, round_name, "MISSING FILE", 0, 0))
            continue

        parsed = parse_xvg(xvg_file)

        if parsed is None:
            DATA[key] = None
            report.append((system, round_name, "NO DATA ROWS", 0, 0))
            continue

        # optional filter
        if HIDE_SERIES_BELOW is not None:
            keep = [
                j for j in range(parsed["n_series"])
                if np.nanmax(parsed["y"][:, j]) >= HIDE_SERIES_BELOW
            ]
            parsed["y"] = parsed["y"][:, keep]
            parsed["labels"] = [parsed["labels"][j] for j in keep]

        DATA[key] = parsed

        for lab in parsed["labels"]:
            get_style(lab)          # register in first-appearance order

        status = "ok"
        if parsed["n_legends"] < parsed["n_series"]:
            status = (f"{parsed['n_series'] - parsed['n_legends']} "
                      f"series without a legend line")
        if parsed["blocks"]:
            status += f"; {parsed['blocks']} '&' block separators found"

        report.append((system, round_name, status,
                       parsed["n_series"], parsed["n_legends"]))


# ============================================================
# HELPERS
# ============================================================

def nice_step(xmax):
    if not np.isfinite(xmax) or xmax <= 0:
        return 1
    raw = xmax / 5.0
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def apply_xticks(ax, x):
    xmax = float(np.nanmax(x))
    xmin = float(np.nanmin(x))
    step = X_TICK_STEP if X_TICK_STEP else nice_step(xmax)
    ticks = np.arange(0, xmax + step * 0.5, step)
    ax.set_xticks(ticks)
    ax.set_xlim(xmin, xmax)


def legend_ncol(n, max_rows=MAX_LEGEND_ROWS):
    return max(1, math.ceil(n / max_rows))


def draw_panel(ax, key, fs, title, fallback_x=None):
    """Draw one system/round panel. Returns list of labels plotted."""
    parsed = DATA.get(key)
    plotted_labels = []

    if parsed is not None and parsed["n_series"] > 0:
        for j, lab in enumerate(parsed["labels"]):
            st = get_style(lab)
            ax.plot(
                parsed["x"], parsed["y"][:, j],
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=1.8,
                alpha=0.95,
                zorder=2,
                label=lab,
            )
            plotted_labels.append(lab)
        apply_xticks(ax, parsed["x"])
    else:
        ax.text(0.5, 0.5, "No hydrogen-bond data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=fs["tick"])
        # keep the empty panel on the same x scale as its neighbours
        if fallback_x is not None:
            apply_xticks(ax, fallback_x)

    # threshold line - labelled, and drawn on top of the traces
    ax.axhline(
        THRESHOLD,
        color="black",
        linestyle=(0, (5, 3)),
        linewidth=2.0,
        alpha=0.95,
        label=THRESHOLD_LABEL,
        zorder=6,
    )

    ax.set_ylim(*Y_LIM)
    ax.set_title(title, fontsize=fs["title"], pad=12)
    ax.tick_params(axis="both", which="major",
                   labelsize=fs["tick"], width=1.5, length=7)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)

    return plotted_labels


# ============================================================
# PASS 2 - INDIVIDUAL FIGURES (ONE PER SYSTEM x ROUND)
# ============================================================

for system in SYSTEMS:

    print("\n======================================")
    print(system)
    print("======================================")

    for round_name in ROUNDS:

        key = (system, round_name)
        fig, ax = plt.subplots(figsize=(15, 9))

        labels_here = draw_panel(
            ax, key, FS_IND,
            panel_title(system, round_name)
        )

        ax.set_xlabel("Frames", fontsize=FS_IND["label"], labelpad=8)
        ax.set_ylabel("Occupancy (%)", fontsize=FS_IND["label"], labelpad=8)

        # ---- legend: every series + the threshold line ----
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.00),
                ncol=legend_ncol(len(labels)),
                fontsize=FS_IND["legend"],
                frameon=True,
                framealpha=1.0,
                borderpad=0.8,
                labelspacing=0.6,
                handlelength=3.0,
                handletextpad=0.8,
                title="Hydrogen bonds",
                title_fontsize=FS_IND["legend"] + 2,
            )

        fig.subplots_adjust(left=0.10, right=0.70, bottom=0.14, top=0.90)

        out_file = os.path.join(
            OUTPUT_DIR, f"{system}_{round_name}_HB_occupancy.png"
        )
        fig.savefig(out_file, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"Saved: {out_file}  ({len(labels_here)} series)")


# ============================================================
# PASS 3 - COMBINED FIGURE
#   rows = systems, column 1 = Round1, column 2 = Round2
# ============================================================

n_rows = len(SYSTEMS)
n_cols = len(ROUNDS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(8.0 * n_cols, 4.2 * n_rows),
    squeeze=False,
    sharex=False,
    sharey=True,
)

used_labels = []

for r, system in enumerate(SYSTEMS):

    # x range of any round of this system, used for empty panels
    fallback_x = None
    for rn in ROUNDS:
        p = DATA.get((system, rn))
        if p is not None:
            fallback_x = p["x"]
            break

    for c, round_name in enumerate(ROUNDS):

        ax = axes[r][c]
        labs = draw_panel(
            ax, (system, round_name), FS_COMB,
            panel_title(system, round_name),
            fallback_x=fallback_x,
        )
        for lab in labs:
            if lab not in used_labels:
                used_labels.append(lab)

        if r == n_rows - 1:
            ax.set_xlabel("Frames", fontsize=FS_COMB["label"], labelpad=6)
        if c == 0:
            ax.set_ylabel("Occupancy (%)", fontsize=FS_COMB["label"],
                          labelpad=6)

# ---- one shared legend for the whole figure ----
handles = [
    Line2D([0], [0],
           color=get_style(lab)["color"],
           linestyle=get_style(lab)["linestyle"],
           linewidth=2.2)
    for lab in used_labels
]
labels = list(used_labels)

handles.append(
    Line2D([0], [0], color="black", linestyle=(0, (5, 3)), linewidth=2.2)
)
labels.append(THRESHOLD_LABEL)

ncol = 3

sup = fig.suptitle(
    "Hydrogen-bond occupancy: R1 (left) vs R2 (right)",
    fontsize=FS_COMB["title"] + 6, y=0.997,
)

# lay the grid out first, then hang the legend underneath it
fig.tight_layout(rect=[0, 0, 1, 0.985])

leg = fig.legend(
    handles, labels,
    loc="upper center",              # legend grows DOWNWARD from the anchor
    bbox_to_anchor=(0.5, 0.0),       # i.e. just below the axes grid
    ncol=ncol,
    fontsize=FS_COMB["legend"],
    frameon=True,
    framealpha=1.0,
    handlelength=3.0,
    labelspacing=0.7,
    columnspacing=1.6,
    title="Hydrogen bonds",
    title_fontsize=FS_COMB["legend"] + 2,
)

combined_path = os.path.join(OUTPUT_DIR, COMBINED_FILE)
# NOTE: once bbox_extra_artists is given, the tight bbox is built from the
# axes + these artists only, so the suptitle has to be listed too or it
# gets cropped off the top of the saved image.
fig.savefig(combined_path, dpi=COMBINED_DPI, bbox_inches="tight",
            bbox_extra_artists=[leg, sup], facecolor="white")
plt.close(fig)

print(f"\nSaved combined figure: {combined_path}")


# ============================================================
# DIAGNOSTIC REPORT - why legends were incomplete
# ============================================================

print("\n======================================")
print("LEGEND / DATA REPORT")
print("======================================")
print(f"{'System':<8}{'Round':<9}{'Cols':>6}{'Legends':>9}  Status")
for system, round_name, status, n_series, n_legends in report:
    print(f"{system:<8}{round_name:<9}{n_series:>6}{n_legends:>9}  {status}")

print(f"\nFont used: {FONT}")
print(f"Unique interactions styled: {len(interaction_styles)} "
      f"(unique colour+linestyle combos available: "
      f"{len(PALETTE) * len(LINESTYLES)})")
print(f"Output directory: {OUTPUT_DIR}")
print("ALL GRAPHS GENERATED")