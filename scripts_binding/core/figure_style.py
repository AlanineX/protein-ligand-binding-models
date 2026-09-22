"""Shared figure style for binding single-panel plots."""

FIGSIZE = (5.0, 3.2)
BASE_FONT_SIZE = 12
AXIS_LABEL_SIZE = BASE_FONT_SIZE + 2
TICK_SIZE = BASE_FONT_SIZE
LEGEND_SIZE = BASE_FONT_SIZE - 2
LINEWIDTH = 1.5
LINE_ALPHA = 0.65
LEGEND_EDGE_COLOR = "#595959"
LEGEND_EDGE_LINEWIDTH = 0.8


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
