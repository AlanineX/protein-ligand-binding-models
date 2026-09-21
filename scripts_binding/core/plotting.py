"""Fit curves, convergence, summary, deconv bar plots."""
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .figure_style import (
    FIGSIZE as SINGLE_FIGSIZE,
    BASE_FONT_SIZE as SINGLE_BASE_FONT_SIZE,
    AXIS_LABEL_SIZE as SINGLE_AXIS_LABEL_SIZE,
    TICK_SIZE as SINGLE_TICK_SIZE,
    LEGEND_SIZE as SINGLE_LEGEND_SIZE,
    LINEWIDTH as SINGLE_LINEWIDTH,
    LINE_ALPHA as SINGLE_LINE_ALPHA,
    transparent_legend_frame,
)


GOLDEN = 1.618  # Subplot/figure width:height ratio for publication aesthetics.

# Fit-curve sizing — same single-panel family as demo and saturation figures.
FIT_BASE_FONT_SIZE = SINGLE_BASE_FONT_SIZE
FIT_AXIS_LABEL_SIZE = SINGLE_AXIS_LABEL_SIZE
FIT_TICK_SIZE = SINGLE_TICK_SIZE
FIT_LEGEND_SIZE = SINGLE_LEGEND_SIZE
FIT_FIGSIZE = SINGLE_FIGSIZE
FIT_LINEWIDTH = SINGLE_LINEWIDTH
FIT_LINE_ALPHA = SINGLE_LINE_ALPHA


def golden_figsize(ncols=1, nrows=1, panel_width=6.0):
    """(w, h) giving each subplot a golden-ratio (1.618:1) aspect."""
    return (panel_width * ncols, (panel_width / GOLDEN) * nrows)


@dataclass
class PlotConfig:
    """Minimal plot-only config shared by the production pipeline and visualize scripts.

    `RunConfig` (core/config.py) is a superset — its plot-related fields match
    these defaults and names, so `RunConfig` instances can be passed to any
    plotting function that expects a `PlotConfig`.

    subplot_box_aspect fixes the plot-area height:width per subplot so every
    panel looks the same regardless of label/title overhead. Default = 1/GOLDEN
    (0.618), giving a slightly elongated panel that matches golden-ratio.
    """
    base_fontsize: int = 20
    output_unit: str = "uM"
    scale_m_to_out: float = 1e6
    deconv_legend_loc: str = "best"
    save_plots: bool = True
    show_plots: bool = False
    max_image_dim: int = 2400
    summary_show_calc_shade: bool = True
    subplot_box_aspect: float = 1.0 / GOLDEN  # plot-area h/w; elongated ~0.45
    colormap: str = "PRGn"
    specific_colormap: str | None = None
    nonspecific_colormap: str | None = None


def setup_matplotlib(cfg):
    """Apply font settings from cfg (Arial house style; Liberation Sans is the
    Arial-metric-compatible fallback, DejaVu Sans the last resort)."""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': cfg.base_fontsize,
    })


def safe_savefig(fig, path, max_image_dim, **kwargs):
    """Save fig capped so no dimension exceeds max_image_dim px."""
    w_in, h_in = fig.get_size_inches()
    max_in = max(w_in, h_in)
    dpi = min(150, max_image_dim / max_in)
    fig.savefig(path, dpi=dpi, **kwargs)


def sized_fig(ncols, nrows,
              cell_w=4.0, cell_h=None,
              left=1.0, right=0.3, top=0.5, bottom=0.9,
              gap_w=0.6, gap_h=1.1):
    """Create a figure where every cell is EXACTLY cell_w × cell_h inches.

    Returns (fig, axes, positions). `positions` is a list of [x, y, w, h] in
    figure fraction coordinates — one per cell in row-major order. Call
    `enforce_positions(axes, positions)` after plotting to guarantee every
    axes has identical dimensions (immune to tight_layout side-effects).

    All sizes in inches. Margins/gaps tuned so titles + rotated xticklabels
    fit inside each cell without overlapping neighbours.
    """
    if cell_h is None:
        cell_h = cell_w / GOLDEN  # golden ratio per cell
    fig_w = left + right + ncols * cell_w + (ncols - 1) * gap_w
    fig_h = top + bottom + nrows * cell_h + (nrows - 1) * gap_h
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)
    # Convert inches → figure fractions
    ax_w = cell_w / fig_w
    ax_h = cell_h / fig_h
    positions = []
    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        x = (left + c * (cell_w + gap_w)) / fig_w
        y = 1 - (top + (r + 1) * cell_h + r * gap_h) / fig_h
        positions.append([x, y, ax_w, ax_h])
    return fig, axes, positions


def enforce_positions(axes, positions):
    """Set each axes to its designated position (run AFTER all plotting)."""
    for ax_row in axes:
        pass  # axes is a 2D list — positions indexed row-major
    flat = [axes[r][c] for r in range(len(axes)) for c in range(len(axes[0]))]
    for ax, pos in zip(flat, positions):
        ax.set_position(pos)


def _diverging_colors(n, cmap_name='PRGn'):
    """n colours from a diverging cmap, skipping near-white centre and near-black ends."""
    cmap = plt.get_cmap(cmap_name)
    if n == 1:
        return [cmap(0.15)]
    t_vals = np.linspace(0, 0.55, n)   # total usable range = 0.25 + 0.30 = 0.55
    colors = []
    for t in t_vals:
        if t <= 0.25:
            colors.append(cmap(0.15 + t))   # [0.15, 0.4]
        else:
            colors.append(cmap(0.35 + t))   # [0.6, 0.9]
    return colors


def _single_hue_colors(n, cmap_name):
    """n visible colors from a single-hue cmap, light apo -> dark high occupancy."""
    cmap = plt.get_cmap(cmap_name)
    if n == 1:
        return [cmap(0.75)]
    return [cmap(0.35 + 0.55 * t) for t in np.linspace(0, 1, n)]


def _species_colors(num_species, cfg, n_specific=None):
    cmap_name = (getattr(cfg, "specific_colormap", None) if n_specific is None else None)
    cmap_name = cmap_name or getattr(cfg, "colormap", "PRGn") or "PRGn"
    if cmap_name == "PRGn":
        return _diverging_colors(num_species)[::-1]
    return _single_hue_colors(num_species, cmap_name)


def _deconv_color_by_count(j, i, S, N, cfg=None, cmap_name='PRGn'):
    """Colour for a (j spec, m = i-j NSB) stack cell.

    Explicit specific/nonspecific maps take priority for the corresponding
    component. An explicit overall map colors both components. With no explicit
    map, the default PRGn map uses the legacy diverging rule:

    Binary hue rule:
        m == 0 (no NSB, pure specific) → GREEN, saturation = j / S
        m >  0 (any NSB present)       → PURPLE, saturation = m / (S + N)

    Every cell with any NSB is purple; only m=0 cells are green. This
    matches the biophysical intuition that NSB contamination — even a
    single ligand — changes the species identity. The returned colour is
    cmap(0.5 ± (0.12 + 0.30·saturation)); the 0.12 floor keeps the lightest
    cells visibly tinted rather than white.

    Example (S = 7, N = 3, so S + N = 10):
        1S+0N → pale green  (j=1 / S=7   = 0.14)
        7S+0N → deep green  (j=7 / S=7   = 1.0)
        0S+1N → pale purple (m=1 / 10    = 0.10)
        0S+10N → deep purple (m=10 / 10  = 1.0)
        6S+1N → pale purple (m=1 / 10    = 0.10, regardless of j=6)
        4S+3N → mid purple  (m=3 / 10    = 0.30)
    """
    chosen_cmap = getattr(cfg, "colormap", "PRGn") or "PRGn"
    specific_cmap = getattr(cfg, "specific_colormap", None)
    nonspecific_cmap = getattr(cfg, "nonspecific_colormap", None)
    if chosen_cmap != "PRGn" or specific_cmap or nonspecific_cmap:
        m = i - j
        name = (specific_cmap if m == 0 else nonspecific_cmap) or chosen_cmap or cmap_name
        cmap = plt.get_cmap(name)
        if i == 0:
            return cmap(0.30)
        max_count = S if m == 0 else S + N
        count = j if m == 0 else m
        sat = min(count / max(max_count, 1), 1.0)
        return cmap(0.30 + 0.60 * sat)

    cmap = plt.get_cmap(cmap_name)
    if i == 0:
        return cmap(0.5)
    m = i - j
    # Depth = count / max-possible-count for that hue, so the gradient spans the
    # full stoichiometry without clamping (max specific = S, max nonspecific =
    # S + N, the largest m at j=0 of the top peak). A small floor keeps the
    # lightest (m=1 / j=1) cells visibly tinted rather than white.
    n_max = S + N
    if m == 0:
        sat = min(j / max(S, 1), 1.0)
        return cmap(0.5 + (0.12 + 0.30 * sat))
    else:
        sat = min(m / max(n_max, 1), 1.0)
        return cmap(0.5 - (0.12 + 0.30 * sat))


def plot_deconv_byconc(
    L_vals_out,
    contrib_stack,
    S,
    N,
    title_prefix,
    cfg,
    outline_totals=None,
    outline_err=None,
    outline_label="Frac_expt",
    shared_legend=False,
):
    """Deconv plot with x-axis = ligand concentration.

    One subplot per bound state i; each subplot's x-axis spans the titration
    concentrations; bars at each [L] are stacked by (j spec, i-j NSB).

    shared_legend=True: one combined legend in an extra panel at the end;
    otherwise per-panel legends showing each panel's specific components.
    """
    base_fontsize = cfg.base_fontsize
    num_species = S + N + 1
    x = np.arange(len(L_vals_out))

    start_i = 1
    n_panels = num_species - start_i
    ncols = 3 if n_panels > 6 else (2 if n_panels > 1 else 1)
    total_cells = n_panels + (1 if shared_legend else 0)
    nrows = int(np.ceil(total_cells / ncols))
    fig, axes, positions = sized_fig(ncols, nrows)
    shared_handles = {}  # (j, m) → Patch, aggregated across panels if shared_legend

    for idx, i in enumerate(range(start_i, num_species)):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        max_j = min(i, S)
        bottom = np.zeros(len(L_vals_out))
        panel_handles = []
        for j in range(max_j + 1):
            col = _deconv_color_by_count(j, i, S, N, cfg=cfg)
            ax.bar(
                x,
                contrib_stack[:, i, j],
                bottom=bottom,
                width=0.8,
                color=col,
                edgecolor='none',
            )
            label = f"{j}S+{i - j}N"
            panel_handles.append(Patch(facecolor=col, edgecolor='none', label=label))
            shared_handles[(j, i - j)] = Patch(facecolor=col, edgecolor='none', label=label)
            bottom += contrib_stack[:, i, j]
        # Column totals per [L] (= bottom after all stacks) drive legend placement
        col_totals = bottom.copy()

        if outline_totals is not None:
            ax.bar(
                x,
                outline_totals[:, i],
                width=0.8,
                facecolor='none',
                edgecolor='black',
                linewidth=0.8,
            )
            panel_handles.append(
                Patch(facecolor='none', edgecolor='black', label=outline_label))
            if outline_err is not None and outline_err.shape[1] > i:
                ax.errorbar(
                    x,
                    outline_totals[:, i],
                    yerr=outline_err[:, i],
                    fmt='none',
                    ecolor='black',
                    elinewidth=0.8,
                    capsize=2,
                    alpha=0.9
                )
        if not shared_legend:
            # Choose corner based on where data concentrates, add headroom so
            # legend sits cleanly above the bars instead of on top of them.
            # y_top must clear BOTH the stacked model bars AND the experimental
            # outline bar plus its error bar (which can exceed the model total).
            y_top = col_totals.max()
            if outline_totals is not None:
                top_out = outline_totals[:, i].astype(float)
                if outline_err is not None and outline_err.shape[1] > i:
                    top_out = top_out + outline_err[:, i]
                y_top = max(y_top, float(np.nanmax(top_out)))
            if y_top > 0:
                denom = col_totals.sum()
                center_of_mass = (col_totals * x).sum() / denom if denom > 0 else len(x) / 2
                # Data left-leaning → legend upper-right (leave right side clear);
                # right-leaning → upper-left (leave left side clear)
                loc = 'upper right' if center_of_mass < len(x) / 2 else 'upper left'
                # Need ~40% extra room for an 8-entry 2-col legend at base*0.5 font
                n_rows_leg = int(np.ceil(len(panel_handles) / 2))
                headroom = 1.0 + 0.10 * n_rows_leg
                ax.set_ylim(0, y_top * headroom)
            else:
                loc = 'upper right'
            leg = ax.legend(
                handles=panel_handles,
                fontsize=base_fontsize * 0.5,
                ncol=2,
                loc=loc,
                frameon=True,
                handlelength=1.0,
                handletextpad=0.35,
                columnspacing=0.7,
            )
            transparent_legend_frame(leg)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.3g}" for v in L_vals_out], rotation=45, ha='right', fontsize=base_fontsize * 0.6)
        ax.tick_params(axis='y', labelsize=base_fontsize * 0.6)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    if shared_legend:
        # Sort aggregated handles by color gradient (deep green → deep purple)
        items = sorted(shared_handles.items(),
                       key=lambda kv: (-kv[0][0] / max(S, 1) if kv[0][1] == 0
                                       else kv[0][1] / max(N, 1)))
        legend_handles = [p for (_, p) in items]
        if outline_totals is not None:
            legend_handles.append(Patch(facecolor='none', edgecolor='black', label=outline_label))
        leg_cell = n_panels
        r_leg, c_leg = divmod(leg_cell, ncols)
        ax_leg = axes[r_leg][c_leg]
        ax_leg.axis('off')
        # Adaptive cols + font so the legend always fits inside the cell
        n = len(legend_handles)
        if n <= 8:
            ncol_leg, fs_mult = 1, 0.55
        elif n <= 16:
            ncol_leg, fs_mult = 2, 0.45
        elif n <= 28:
            ncol_leg, fs_mult = 3, 0.38
        else:
            ncol_leg, fs_mult = 4, 0.32
        leg = ax_leg.legend(handles=legend_handles, loc='center',
                            fontsize=base_fontsize * fs_mult, ncol=ncol_leg,
                            frameon=True, handlelength=0.8, handletextpad=0.3,
                            columnspacing=0.6, labelspacing=0.25)
        transparent_legend_frame(leg)
        used_cells = n_panels + 1
    else:
        used_cells = n_panels
    for cell in range(used_cells, nrows * ncols):
        r, c = divmod(cell, ncols)
        axes[r][c].axis('off')

    fig.supxlabel(f"Total Ligand Concentration ({cfg.output_unit})", fontsize=base_fontsize * 0.85)
    fig.supylabel("Fraction", fontsize=base_fontsize * 0.85)
    enforce_positions(axes, positions)
    return fig


def plot_deconv_byligand(
    L_vals_out,
    contrib_stack,
    S,
    N,
    title_prefix,
    cfg,
    outline_totals=None,
    outline_err=None,
    outline_label="Frac_expt",
    shared_legend=False,
):
    """Deconv plot with x-axis = ligand-bound state index I_i.

    One subplot per ligand concentration; x-axis spans I_0..I_{S+N}; each
    bar is stacked by the (j spec, i-j NSB) decomposition. Complements
    `plot_deconv_byconc` (which flips the axes).

    shared_legend=True: one combined legend in an extra panel at the end;
    otherwise per-panel legends showing the significant components.
    """
    base_fontsize = cfg.base_fontsize
    num_species = S + N + 1
    x = np.arange(num_species)

    n_panels = len(L_vals_out)
    ncols = 3 if n_panels > 6 else (2 if n_panels > 1 else 1)
    total_cells = n_panels + (1 if shared_legend else 0)
    nrows = int(np.ceil(total_cells / ncols))
    fig, axes, positions = sized_fig(ncols, nrows)
    shared_handles = {}

    for idx, L in enumerate(L_vals_out):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        seen = {}
        bar_totals = np.zeros(num_species)  # total height per I_i bar, for legend placement
        for i in range(num_species):
            max_j = min(i, S)
            bottom = 0.0
            for j in range(max_j + 1):
                v = contrib_stack[idx, i, j]
                if v <= 0:
                    continue
                col = _deconv_color_by_count(j, i, S, N, cfg=cfg)
                ax.bar(i, v, bottom=bottom, width=0.8, color=col, edgecolor='none')
                bottom += v
                m = i - j
                key = (j, m)
                if key not in seen or v > seen[key][1]:
                    seen[key] = (col, v)
                if v > 0.005:
                    shared_handles[key] = Patch(facecolor=col, edgecolor='none', label=f"{j}S+{m}N")
            bar_totals[i] = bottom
        if outline_totals is not None:
            ax.bar(x, outline_totals[idx], width=0.8,
                   facecolor='none', edgecolor='black', linewidth=0.6)
            if outline_err is not None and outline_err.shape[1] > 0:
                ax.errorbar(x, outline_totals[idx], yerr=outline_err[idx],
                            fmt='none', ecolor='black', elinewidth=0.8, capsize=2, alpha=0.9)

        if not shared_legend:
            sig = [(k, v) for k, v in seen.items() if v[1] > 0.005]
            sig.sort(key=lambda item: (-item[0][0] / max(S, 1) if item[0][1] == 0
                                       else item[0][1] / max(N, 1)))
            panel_handles = [
                Patch(facecolor=col, edgecolor='none', label=f"{j}S+{m}N")
                for (j, m), (col, _) in sig
            ]
            if outline_totals is not None:
                panel_handles.append(
                    Patch(facecolor='none', edgecolor='black', label=outline_label))
            # Place legend in upper-left or upper-right based on where bars peak;
            # add headroom so the legend sits above the data, not on top of it.
            if bar_totals.max() > 0:
                center_of_mass = (bar_totals * np.arange(num_species)).sum() / bar_totals.sum()
                loc = 'upper right' if center_of_mass < num_species / 2 else 'upper left'
                ncol_leg = 2 if len(panel_handles) <= 10 else 3
                n_rows_leg = int(np.ceil(len(panel_handles) / ncol_leg))
                headroom = 1.0 + 0.09 * n_rows_leg
                ax.set_ylim(0, bar_totals.max() * headroom)
            else:
                loc = 'upper right'
                ncol_leg = 2
            leg = ax.legend(
                handles=panel_handles,
                fontsize=base_fontsize * 0.5,
                ncol=ncol_leg,
                loc=loc,
                frameon=True,
                handlelength=1.0,
                handletextpad=0.35,
                columnspacing=0.7,
            )
            transparent_legend_frame(leg)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$I_{{{ix}}}$" for ix in x], fontsize=base_fontsize * 0.6)
        ax.tick_params(axis='y', labelsize=base_fontsize * 0.6)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    if shared_legend:
        items = sorted(shared_handles.items(),
                       key=lambda kv: (-kv[0][0] / max(S, 1) if kv[0][1] == 0
                                       else kv[0][1] / max(N, 1)))
        legend_handles = [p for (_, p) in items]
        if outline_totals is not None:
            legend_handles.append(Patch(facecolor='none', edgecolor='black', label=outline_label))
        leg_cell = n_panels
        r_leg, c_leg = divmod(leg_cell, ncols)
        ax_leg = axes[r_leg][c_leg]
        ax_leg.axis('off')
        # Adaptive cols + font so the legend always fits inside the cell
        n = len(legend_handles)
        if n <= 8:
            ncol_leg, fs_mult = 1, 0.55
        elif n <= 16:
            ncol_leg, fs_mult = 2, 0.45
        elif n <= 28:
            ncol_leg, fs_mult = 3, 0.38
        else:
            ncol_leg, fs_mult = 4, 0.32
        leg = ax_leg.legend(handles=legend_handles, loc='center',
                            fontsize=base_fontsize * fs_mult, ncol=ncol_leg,
                            frameon=True, handlelength=0.8, handletextpad=0.3,
                            columnspacing=0.6, labelspacing=0.25)
        transparent_legend_frame(leg)
        used_cells = n_panels + 1
    else:
        used_cells = n_panels
    for cell in range(used_cells, nrows * ncols):
        r, c = divmod(cell, ncols)
        axes[r][c].axis('off')

    fig.supxlabel("Bound state", fontsize=base_fontsize * 0.85)
    fig.supylabel("Fraction", fontsize=base_fontsize * 0.85)
    enforce_positions(axes, positions)
    return fig


def plot_species_curves(L_grid_M, F_grid, num_species, cfg, *, output_svg=None,
                        title=None, n_specific=None, data_L_M=None, data_F=None,
                        data_err=None, calc_std=None, ax=None, colors=None,
                        model_species_count=None, x_tick_rotation=None,
                        show_ladder_markers=False, legend_kd_values=None,
                        legend_kd_unit=None):
    """THE single species-curve plot — model curves (lines) + optional data points.

    Serves per-replicate fits, replicate-averaged summaries, and simulations with
    one consistent style: FIT_BASE_FONT_SIZE/FIT_FIGSIZE, `_species_label`
    legend, 'o' markers in the species colour (with error bars when `data_err`
    is given), `[L]` axis, gridlines at the data concentrations, 0.2 y-ticks.

    L_grid_M, F_grid : fine model curves, shape (nL, num_species); L in molar.
    model_species_count : number of species actually generated by the model.
        Observed higher species can still be shown as data points, but padded
        zero model curves are not drawn.
    data_L_M (nconc,), data_F (nconc,k) : experimental points; data_F=None => lines
        only (e.g. simulations). data_err (nconc,k) => error bars.
    calc_std : optional ±std band on the model curves (needs cfg.summary_show_calc_shade).
    n_specific : curves with index > n_specific are dashed (NSB); None => all solid.
    output_svg : save+close when given; otherwise return (fig, ax) for overlays
        (e.g. drawing composition pies on top).
    """
    # AMAC/EDDA summaries use condition palettes; caller may override.
    if colors is None:
        if legend_kd_values is not None and num_species > 1:
            # Kd-annotated plots map the full colormap directly onto PL1..PLn.
            # Apo P is neutral because it has no associated dissociation step.
            colors = _species_colors(num_species, cfg, n_specific=n_specific)
            colors[0] = (0.40, 0.40, 0.40, 1.0)
        else:
            colors = _species_colors(num_species, cfg, n_specific=n_specific)
    own = ax is None
    fig, ax = plt.subplots(figsize=FIT_FIGSIZE) if own else (ax.figure, ax)
    sm = cfg.scale_m_to_out
    if model_species_count is None:
        model_species_count = F_grid.shape[1]
    for j in range(num_species):
        if j < model_species_count:
            ls = '--' if (n_specific is not None and j > n_specific) else '-'
            species_label = _species_label(j)
            if legend_kd_values is not None and 0 < j <= len(legend_kd_values):
                kd = float(legend_kd_values[j - 1])
                if np.isfinite(kd):
                    unit = (legend_kd_unit or cfg.output_unit).replace('uM', r'$\mu$M')
                    species_label += rf" ($K_{{d,{j}}}$ = {_format_legend_value(kd)} {unit})"
            ax.plot(L_grid_M * sm, F_grid[:, j], lw=FIT_LINEWIDTH, color=colors[j],
                    alpha=FIT_LINE_ALPHA, linestyle=ls, label=species_label)
        if j < model_species_count and calc_std is not None and getattr(cfg, 'summary_show_calc_shade', False):
            ax.fill_between(L_grid_M * sm, F_grid[:, j] - calc_std[:, j],
                            F_grid[:, j] + calc_std[:, j], color=colors[j],
                            alpha=0.15, linewidth=0)
        if data_F is not None and j < data_F.shape[1]:
            mask = ~np.isnan(data_F[:, j])
            yerr = data_err[mask, j] if (data_err is not None and data_err.shape[1] > j) else None
            ax.errorbar(np.asarray(data_L_M)[mask] * sm, data_F[mask, j], yerr=yerr,
                        fmt='o', ms=5, color=colors[j], ecolor=colors[j],
                        capsize=3, alpha=0.8, zorder=3)
        if (show_ladder_markers and data_F is None and data_L_M is not None
                and j < model_species_count):
            ladder_L = np.asarray(data_L_M, dtype=float)
            in_range = (ladder_L >= np.nanmin(L_grid_M)) & (ladder_L <= np.nanmax(L_grid_M))
            if np.any(in_range):
                ladder_y = np.interp(ladder_L[in_range], L_grid_M, F_grid[:, j])
                ax.plot(ladder_L[in_range] * sm, ladder_y, linestyle='None', marker='o',
                        ms=4.5, color=colors[j], markerfacecolor=colors[j],
                        markeredgecolor='white', markeredgewidth=0.45,
                        alpha=0.95, label='_nolegend_', zorder=4)

    ax.set_xlabel(f"[L] ({cfg.output_unit.replace('uM', 'µM')})", fontsize=FIT_AXIS_LABEL_SIZE)
    ax.set_ylabel('Mole Fraction', fontsize=FIT_AXIS_LABEL_SIZE)
    xt = data_L_M if data_L_M is not None else L_grid_M
    ax.set_xticks(np.unique(np.asarray(xt) * sm))
    if x_tick_rotation is not None:
        for tick_label in ax.get_xticklabels():
            tick_label.set_rotation(float(x_tick_rotation))
            tick_label.set_ha('right')
            tick_label.set_rotation_mode('anchor')
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(labelsize=FIT_TICK_SIZE)
    ax.set_ylim(0, 1.02)
    outside_legend = legend_kd_values is not None or num_species > 6
    if legend_kd_values is not None:
        leg = ax.legend(
            ncol=1, loc="upper left", bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0, fontsize=FIT_LEGEND_SIZE, frameon=True,
        )
        leg.set_in_layout(False)
    elif num_species > 6:
        leg = ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0, fontsize=FIT_LEGEND_SIZE, frameon=True)
        leg.set_in_layout(False)
    else:
        leg = ax.legend(ncol=2, loc="upper right", fontsize=FIT_LEGEND_SIZE,
                        frameon=True)
    transparent_legend_frame(leg)
    ax.grid(True, linestyle="--", alpha=0.6)
    if own:
        if outside_legend:
            # Preserve the standard binding-panel axes width and add a
            # dedicated right-side column for the expanded vertical legend.
            # Manual margins avoid tight_layout shrinking the data panel to
            # accommodate an artist that intentionally sits outside the axes.
            fig.set_size_inches(FIT_FIGSIZE[0] + 3.0, FIT_FIGSIZE[1], forward=True)
            fig.subplots_adjust(left=0.10, right=0.62, bottom=0.20, top=0.96)
        else:
            fig.tight_layout()
    if output_svg is not None:
        if getattr(cfg, 'save_plots', True):
            safe_savefig(fig, output_svg, cfg.max_image_dim)
        if getattr(cfg, 'show_plots', False):
            plt.show()
        else:
            plt.close(fig)
        return None
    return fig, ax


def _species_label(j):
    """P (j=0, apo) / PL (j=1) / PL_k (j≥2) with subscript formatting."""
    if j == 0:
        return "$P$"
    if j == 1:
        return "$PL$"
    return f"$PL_{{{j}}}$"


def _format_legend_value(value):
    """Compact fitted-value formatting suitable for a figure legend."""
    value = float(value)
    magnitude = abs(value)
    if magnitude != 0 and (magnitude >= 1e4 or magnitude < 1e-2):
        return f"{value:.2e}".replace("e+", "e")
    return f"{value:.2f}".rstrip("0").rstrip(".")


# NOTE: per-replicate fits, replicate-averaged summaries, and simulations all use
# the single `plot_species_curves` above. (The former plot_fit_curves /
# plot_summary_fit were merged into it.)
