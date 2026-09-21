"""Shared figure style for binding single-panel plots."""

FIGSIZE = (5.0, 3.2)
BASE_FONT_SIZE = 12
AXIS_LABEL_SIZE = BASE_FONT_SIZE + 2
TICK_SIZE = BASE_FONT_SIZE
ANNOT_SIZE = BASE_FONT_SIZE - 1
LEGEND_SIZE = BASE_FONT_SIZE - 2
LINEWIDTH = 1.5
LINE_ALPHA = 0.65
DPI = 600
LEGEND_EDGE_COLOR = "#595959"
LEGEND_EDGE_LINEWIDTH = 0.8
GRID_COLOR = "0.72"
GRID_LINESTYLE = ":"
GRID_LINEWIDTH = 0.45
GRID_ALPHA = 0.38


def apply_font_rc(plt, *, font_size=BASE_FONT_SIZE):
    """Apply the shared font family and math-font defaults."""
    plt.rcParams.update({
        "font.size": font_size,
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })


def savefig_kwargs():
    """Save settings for the 5 x 3.2 in white-canvas figure family."""
    return {"dpi": DPI, "transparent": False, "facecolor": "white"}


def transparent_legend_frame(leg, edgecolor=LEGEND_EDGE_COLOR,
                             linewidth=LEGEND_EDGE_LINEWIDTH):
    """Keep a legend border while removing the fill in raster/vector output."""
    if leg is None:
        return None
    frame = leg.get_frame()
    frame.set_fill(False)
    frame.set_facecolor("none")
    frame.set_edgecolor(edgecolor)
    frame.set_linewidth(linewidth)
    frame.set_alpha(1.0)
    return leg


def apply_binding_grid(ax):
    """Apply the soft grid style for compact binding single-panel plots."""
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color=GRID_COLOR, linestyle=GRID_LINESTYLE,
            linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
