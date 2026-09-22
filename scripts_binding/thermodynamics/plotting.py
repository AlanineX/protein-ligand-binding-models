"""Van't Hoff and thermodynamics plots — built on core.plotting primitives.

Reuses PlotConfig (font defaults), golden_figsize, _diverging_colors, and
safe_savefig so these plots look visually consistent with the Kd-side deconv
and mole-fraction plots.
"""
import colorsys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from ..core.plotting import (
    PlotConfig,
    _diverging_colors,
    golden_figsize,
    safe_savefig,
)
from .nlvh import lvh_equation, nlvh_equation


def _muted(hex_color, sat_scale=0.85, l_shift=0.02):
    """Adjust saturation × sat_scale and lightness by +l_shift (HLS space)."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0, min(1, s * sat_scale))
    l = max(0, min(1, l + l_shift))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))


# Thermodynamics palettes. Each triple is ordered as (ΔG, ΔH, −TΔS).
PALETTE_PRESETS = {
    # Matplotlib tab10 — most commonly seen in publications
    "tab":       ("tab:orange",     "tab:green",       "tab:purple"),
    # CSS classics — bright, unambiguous
    "classic":   ("orange",         "green",           "purple"),
    # Deep & saturated
    "deep":      ("darkorange",     "darkgreen",       "rebeccapurple"),
    # Rich, slightly earthy
    "rich":      ("chocolate",      "forestgreen",     "darkorchid"),
    # Soft pastel-leaning
    "soft":      ("coral",          "seagreen",        "mediumpurple"),
    # Muted / earthy natural
    "muted":     ("peru",           "olivedrab",       "slateblue"),
    # Warm-cool contrast
    "warm":      ("darkorange",     "green",           "indigo"),
    # Bright vibrant
    "vibrant":   ("orangered",      "limegreen",       "blueviolet"),
    # PRGn-derived: purple=PRGn(0.3), green=PRGn(0.7), orange = mathematical
    # complement of their hue-midpoint (197.8° → 17.8°) with matched L&S (0.73, 0.37)
    "prgn_matched": ("#d4b0a1",     "#a5da9f",         "#c1a4ce"),
    # Same PRGn purple/green but with saturation-boosted complementary orange
    "prgn_boost":   ("#c27657",     "#a5da9f",         "#c1a4ce"),
    # Harmonic palettes: ΔG=gray (recedes), ΔH=K5 green, −TΔS=K2 purple.
    # K2/K5 are the 2nd/5th colors of _diverging_colors(6) on PRGn — same
    # colors used in the 6-site van't Hoff panels.
    "k25_gray_light":  ("#bfbfbf",  "#60b266",         "#b18fc0"),  # light gray
    "k25_gray_medium": ("#a6a6a6",  "#60b266",         "#b18fc0"),  # medium gray
    "k25_gray_dark":   ("#737373",  "#60b266",         "#b18fc0"),  # dark gray
    "k25_slate":       ("slategray","#60b266",         "#b18fc0"),  # blue-gray
    # paper-v2: silver ΔG + PL5 green / PL4 purple from _diverging_colors(n=11),
    # matching the fig1f/1i deconv palette gradient exactly. See
    # PAPER_V2_BAR_STYLE for per-bar alpha/edge/lw.
    "paper-v2":        ("#C0C0C0",  "#89ca88",         "#b18fc0"),
    # Exact RGB samples from Matplotlib's Spectral colormap:
    # ΔG = Spectral(0.80), ΔH = Spectral(0.90), −TΔS = Spectral(0.10).
    "harmonious":      (
        plt.get_cmap("Spectral")(0.80)[:3],
        plt.get_cmap("Spectral")(0.90)[:3],
        plt.get_cmap("Spectral")(0.10)[:3],
    ),
    # C_crisp_focus trial: named colors reconstructed exactly from the raster.
    "crisp-focus":     ("darkgreen", "cornflowerblue", "salmon"),
    # Matplotlib Tableau RGB channels requested for the active thermodynamics plots.
    "tab-rgb":         ("tab:green", "tab:blue", "tab:red"),
}


# Per-bar styling for the paper-v2 palette — silver ΔG (extra-transparent so
# the enthalpy/entropy bars dominate visually), PL5 green (fig1f PL5 fill,
# PL10 outline) and PL4 purple (fig1f PL4 fill, PRGn 0.1 outline).
PAPER_V2_BAR_STYLE = {
    "dG":  {"alpha": 0.25, "edgecolor": "#000000", "lw": 1.25},
    "dH":  {"alpha": 1.0,  "edgecolor": "#89ca88", "lw": 1.5},  # edge = fill (PL7)
    "TdS": {"alpha": 1.0,  "edgecolor": "#b18fc0", "lw": 1.5},  # edge = fill (PL2)
}

HARMONIOUS_BAR_STYLE = {
    "dG":  {"alpha": 0.80, "edgecolor": plt.get_cmap("Spectral")(0.80)[:3], "lw": 1.0},
    "dH":  {"alpha": 0.80, "edgecolor": plt.get_cmap("Spectral")(0.90)[:3], "lw": 1.0},
    "TdS": {"alpha": 0.80, "edgecolor": plt.get_cmap("Spectral")(0.10)[:3], "lw": 1.0},
}

CRISP_FOCUS_BAR_STYLE = {
    "dG":  {"alpha": 0.72, "edgecolor": "darkgreen", "lw": 1.0},
    "dH":  {"alpha": 0.72, "edgecolor": "cornflowerblue", "lw": 1.0},
    "TdS": {"alpha": 0.72, "edgecolor": "salmon", "lw": 1.0},
}

TAB_RGB_BAR_STYLE = {
    "dG":  {"alpha": 0.72, "edgecolor": "black", "lw": 1.5},
    "dH":  {"alpha": 0.72, "edgecolor": "none", "lw": 0.0},
    "TdS": {"alpha": 0.72, "edgecolor": "none", "lw": 0.0},
}


def thermo_palette(name="classic"):
    """Return (ΔG, ΔH, −TΔS) hex triple for a named-color preset.

    See PALETTE_PRESETS for the full list. Every preset follows the
    (ΔG, ΔH, −TΔS) channel order.
    """
    trio = PALETTE_PRESETS[name]
    return tuple(mcolors.to_hex(mcolors.to_rgb(c)) for c in trio)


# Keep ``crisp-focus`` available as the prior palette; use Tableau RGB by default.
DEFAULT_PALETTE = "tab-rgb"
DEFAULT_ALPHA = 0.85

# Default y-axis range for comparable thermo bar plots. If bars/error bars
# extend far outside this range, plot_thermo_bars uses a broken y-axis with
# separate high/mid/low windows instead of stretching one continuous axis.
THERMO_YLIM = (-35, 5)
THERMO_YTICK_MAJOR = 5    # kJ/mol between major gridlines
THERMO_YTICK_MINOR = 1    # kJ/mol between minor gridlines
THERMO_BREAK_GAP = 5      # minimum omitted gap between adjacent y-windows
THERMO_BREAK_MIN_EXCESS = 8
THERMO_BREAK_TARGET_SPAN = 80
THERMO_EDGE_PAD = 1.0     # kJ/mol; avoid SD caps sitting on the plot border
# Thermodynamics bar-plot family sizing.  The height is shared across models;
# the plotting area grows horizontally with the number of K groups.
THERMO_FIG_HEIGHT = 5.4
THERMO_GROUP_WIDTH = 0.9
THERMO_SIDE_WIDTH = 2.2


def _thermo_figsize(n_groups):
    """Return a fixed-height size with width scaled to the K-group count."""
    return (
        THERMO_SIDE_WIDTH + THERMO_GROUP_WIDTH * max(1, int(n_groups)),
        THERMO_FIG_HEIGHT,
    )


def _xtick_labels(sub):
    """Return x-tick labels using math typography.

    Specific-stepwise-only (no Kn present): K_1, K_2, …, K_N
    NSB models (Kn present):                K_{s,1}, …, K_{s,S}, K_n
    """
    import re
    has_kn = any(bool(v) for v in sub["is_kn"])
    labels = []
    for _, row in sub.iterrows():
        if bool(row.is_kn):
            labels.append(r"$K_\mathrm{n}$")
            continue
        # Extract numeric index from labels like 'K1', 'Ks_3', 'K7', etc.
        m = re.search(r"(\d+)", str(row.label))
        idx = m.group(1) if m else str(row.label)
        if has_kn:
            labels.append(rf"$K_{{s,{idx}}}$")
        else:
            labels.append(rf"$K_{{{idx}}}$")
    return labels

COLOR_DG, COLOR_DH, COLOR_TDS = thermo_palette(DEFAULT_PALETTE)


def _bar_style(palette, channel, default_alpha, fill_hex):
    """Per-bar {facecolor, edgecolor, lw} for (palette, channel).

    channel ∈ {"dG", "dH", "TdS"}. Alpha is baked into the facecolor RGBA so
    that `edgecolor` stays at full opacity independent of fill transparency
    (matplotlib's top-level ``alpha`` kwarg would otherwise scale both).
    """
    if palette == "paper-v2":
        s = PAPER_V2_BAR_STYLE[channel]
        a = s["alpha"]
        edge = s["edgecolor"]
        lw = s["lw"]
    elif palette == "harmonious":
        s = HARMONIOUS_BAR_STYLE[channel]
        a = s["alpha"]
        edge = s["edgecolor"]
        lw = s["lw"]
    elif palette == "crisp-focus":
        s = CRISP_FOCUS_BAR_STYLE[channel]
        a = s["alpha"]
        edge = s["edgecolor"]
        lw = s["lw"]
    elif palette == "tab-rgb":
        s = TAB_RGB_BAR_STYLE[channel]
        a = s["alpha"]
        edge = s["edgecolor"]
        lw = s["lw"]
    else:
        a = default_alpha
        edge = "black"
        lw = 0.5
    facecolor = mcolors.to_rgba(fill_hex, a)
    return {"facecolor": facecolor, "edgecolor": edge, "lw": lw}


def plot_vanthoff(
    fits_by_key,
    key,
    output_path,
    cfg=None,
    title="",
    show_errors=True,
    equilibrium_constant="Ka",
):
    """Plot per-site lnKa or lnKd against inverse temperature."""
    cfg = cfg or PlotConfig()
    constant = str(equilibrium_constant).strip().lower()
    if constant not in {"ka", "kd"}:
        raise ValueError("equilibrium_constant must be 'Ka' or 'Kd'")
    show_kd = constant == "kd"
    bundle = fits_by_key[key]
    N, method_fallback = key
    temps_K = bundle["temps_K"]
    lnKa_mean = bundle["lnKa_mean"]
    # Displayed error bars: sample SD across replicates when it exists
    # (consistent with the avg±SD reporting protocol used everywhere else).
    # With a single replicate per temperature the replicate SD is all-NaN,
    # so fall back to the propagated per-point fit SE (σ_Kd/Kd, carried by
    # 'lnKa_std') — the same σ the weighted regression uses — and relabel.
    lnKa_sd = bundle.get("lnKa_sd")
    if lnKa_sd is None or np.all(np.isnan(lnKa_sd)):
        lnKa_std = bundle["lnKa_std"]
        has_display_errors = show_errors and np.any(np.isfinite(lnKa_std) & (lnKa_std > 0))
        err_suffix = " ± SE (fit)" if has_display_errors else ""
    else:
        lnKa_std = lnKa_sd
        has_display_errors = show_errors and np.any(np.isfinite(lnKa_std) & (lnKa_std > 0))
        err_suffix = " mean ± SD" if has_display_errors else ""

    # Need both methods so each panel can show the LVH/NLVH comparison.
    nlvh_rows = {r["site"]: r for r in fits_by_key.get((N, "nlvh"), bundle)["rows"]
                 if r["replicate"] == "mean"}
    lvh_rows = {r["site"]: r for r in fits_by_key.get((N, "lvh"), bundle)["rows"]
                if r["replicate"] == "mean"}
    sites = sorted(set(nlvh_rows) | set(lvh_rows))
    if not sites:
        return None

    n_panels = len(sites)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=golden_figsize(ncols, nrows),
        squeeze=False,
    )
    colors = _diverging_colors(n_panels, getattr(cfg, "colormap", "PRGn"))
    base = cfg.base_fontsize

    for idx, si_num in enumerate(sites):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        nl = nlvh_rows.get(si_num)
        lv = lvh_rows.get(si_num)
        si = si_num - 1
        y = lnKa_mean[:, si]
        if show_kd:
            y = -y
        yerr = lnKa_std[:, si]
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            ax.axis("off")
            continue

        inv_T_data = 1000.0 / temps_K[mask]
        constant_symbol = rf"K_{{{'d' if show_kd else 'a'},{si_num}}}"
        err_label = rf"$\ln({constant_symbol})${err_suffix}"
        ax.errorbar(
            inv_T_data, y[mask], yerr=(yerr[mask] if has_display_errors else None),
            fmt="o", color=colors[idx], ecolor=colors[idx],
            markersize=7, markeredgecolor="black", markeredgewidth=0.5,
            capsize=3, zorder=4, label=err_label,
        )

        T_fine = np.linspace(temps_K[mask].min() - 2, temps_K[mask].max() + 2, 200)
        rec = method_fallback

        # Draw both fit curves; linestyle is tied to the METHOD (solid = LVH,
        # dashed = NLVH) so the reader sees the same shape across every panel.
        # Legend reports each method's R²,
        # so the reader can compare goodness-of-fit directly. Both curves use
        # the same line weight; recommendation is implicit via R².
        if lv is not None:
            y_lv = lvh_equation(T_fine, lv["dH_kJmol"], lv["dS_kJmolK"])
            if show_kd:
                y_lv = -y_lv
            r2_lv = lv.get("R2")
            r2_str = f" ($R^2$ = {r2_lv:.3f})" if r2_lv is not None else ""
            ax.plot(1000.0 / T_fine, y_lv,
                    linestyle="-",
                    lw=2.2,
                    color=colors[idx],
                    alpha=1.0,
                    label=f"LVH{r2_str}",
                    zorder=(3 if rec == "lvh" else 2))
        if nl is not None:
            y_nl = nlvh_equation(T_fine, nl["dH_kJmol"], nl["dS_kJmolK"],
                                  nl["Cp_kJmolK"], nl["T0_K"])
            if show_kd:
                y_nl = -y_nl
            r2_nl = nl.get("R2")
            cp = nl.get("Cp_kJmolK")
            e_cp = nl.get("e_Cp")
            label_parts = ["NLVH"]
            if r2_nl is not None:
                label_parts.append(f"$R^2$ = {r2_nl:.3f}")
            if cp is not None and e_cp is not None and show_errors:
                label_parts.append(f"$\\Delta C_p$ = {cp:+.2f} ± {e_cp:.2f}")
            elif cp is not None:
                label_parts.append(f"$\\Delta C_p$ = {cp:+.2f}")
            label_str = label_parts[0] + " (" + ", ".join(label_parts[1:]) + ")" \
                        if len(label_parts) > 1 else label_parts[0]
            # Use matplotlib's built-in '--' so the legend handle reliably
            # reproduces the dash pattern; the previous tuple linestyle
            # (0, (6, 3)) was rendering as a solid handle in some legend
            # placements.
            ax.plot(1000.0 / T_fine, y_nl,
                    linestyle="--",
                    lw=2.2,
                    color=colors[idx],
                    alpha=1.0,
                    label=label_str,
                    dashes=(5, 3),
                    zorder=(3 if rec == "nlvh" else 2))

        ax.set_xlabel(r"$1000/T$ (K$^{-1}$)", fontsize=base * 0.7)
        ax.set_ylabel(rf"$\ln({constant_symbol})$", fontsize=base * 0.7)
        ax.tick_params(labelsize=base * 0.55)
        # handlelength bumped from default 2.0 → 2.8 so the dashed NLVH
        # handle shows at least one full dash period in the legend box.
        ax.legend(fontsize=base * 0.45, frameon=False, loc="best",
                  handlelength=2.8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_box_aspect(cfg.subplot_box_aspect)

    for cell in range(n_panels, nrows * ncols):
        r, c = divmod(cell, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    safe_savefig(fig, output_path, cfg.max_image_dim)
    plt.close(fig)
    return output_path


def _rounded_floor(x, step=5.0):
    return float(np.floor(float(x) / step) * step)


def _rounded_ceil(x, step=5.0):
    return float(np.ceil(float(x) / step) * step)


def _bar_segments(values, errors, default=THERMO_YLIM):
    """Return top-to-bottom y-axis windows for thermo bar plots.

    The default middle window stays fixed for ordinary plots. Large excursions
    get separate high and/or low windows so the main energy region remains
    readable while extreme bars and error bars are still shown.
    """
    vals = np.asarray(values, dtype=float)
    errs = np.nan_to_num(np.asarray(errors, dtype=float), nan=0.0)
    finite = np.isfinite(vals)
    if not finite.any():
        return [(default[0], default[1], "mid")]
    lo = float(np.min(vals[finite] - errs[finite]))
    hi = float(np.max(vals[finite] + errs[finite]))
    mid_lo, mid_hi = default

    needs_high_break = hi > mid_hi + THERMO_BREAK_MIN_EXCESS
    needs_low_break = lo < mid_lo - THERMO_BREAK_MIN_EXCESS
    if not needs_high_break and not needs_low_break:
        plot_lo = mid_lo
        plot_hi = mid_hi
        if lo < mid_lo + THERMO_EDGE_PAD:
            plot_lo = min(plot_lo, _rounded_floor(lo - THERMO_EDGE_PAD))
        if hi > mid_hi - THERMO_EDGE_PAD:
            plot_hi = max(plot_hi, _rounded_ceil(hi + THERMO_EDGE_PAD))
        return [(plot_lo, plot_hi, "mid")]

    segments = []

    if needs_high_break:
        high_centers = vals[finite & (vals > mid_hi)]
        high_top = _rounded_ceil(hi + 2.0)
        if len(high_centers):
            high_bottom_raw = min(
                float(np.min(high_centers)) - 5.0,
                high_top - THERMO_BREAK_TARGET_SPAN,
            )
        else:
            high_bottom_raw = high_top - THERMO_BREAK_TARGET_SPAN
        high_bottom = max(
            mid_hi + THERMO_BREAK_GAP,
            _rounded_floor(high_bottom_raw),
        )
        if high_top - high_bottom < 20:
            high_top = high_bottom + 20
        segments.append((high_bottom, high_top, "high"))

    segments.append((mid_lo, mid_hi, "mid"))

    if needs_low_break:
        low_centers = vals[finite & (vals < mid_lo)]
        low_bottom = _rounded_floor(lo - 2.0)
        if len(low_centers):
            low_top_raw = max(
                float(np.max(low_centers)) + 5.0,
                low_bottom + THERMO_BREAK_TARGET_SPAN,
            )
        else:
            low_top_raw = low_bottom + THERMO_BREAK_TARGET_SPAN
        low_top = min(
            mid_lo - THERMO_BREAK_GAP,
            _rounded_ceil(low_top_raw),
        )
        if low_top - low_bottom < 20:
            low_bottom = low_top - 20
        segments.append((low_bottom, low_top, "low"))

    return segments


def _segment_height_ratios(segments):
    ratios = []
    for ymin, ymax, kind in segments:
        span = max(float(ymax) - float(ymin), 1.0)
        if kind == "mid":
            ratios.append(1.0)
        else:
            ratios.append(float(np.clip(span / THERMO_BREAK_TARGET_SPAN, 0.8, 1.25)))
    return ratios


def _draw_break_marks(ax_upper, ax_lower, upper_ratio, lower_ratio):
    d = 0.012
    kwargs = {"color": "black", "clip_on": False, "lw": 0.9}
    up_scale = lower_ratio / upper_ratio
    lo_scale = upper_ratio / lower_ratio
    for x in (-0.018, 1.018):
        ax_upper.plot(
            (x - d, x + d),
            (-d * up_scale, d * up_scale),
            transform=ax_upper.transAxes,
            **kwargs,
        )
        ax_lower.plot(
            (x - d, x + d),
            (1 - d * lo_scale, 1 + d * lo_scale),
            transform=ax_lower.transAxes,
            **kwargs,
        )


def _draw_thermo_bars_on_axis(ax, x, sub, width, offset, ec, s_dG, s_dH, s_TdS,
                              show_legend=False, base=20, show_errors=True):
    yerr_dG = sub["e_dG"] if show_errors else None
    yerr_dH = sub["e_dH"] if show_errors else None
    yerr_TdS = sub["e_mTdS"] if show_errors else None
    ax.bar(x - offset, sub["dG_kJmol"], width, yerr=yerr_dG,
           label=r"$\Delta G$", error_kw=ec, zorder=3, **s_dG)
    ax.bar(x, sub["dH_kJmol"], width, yerr=yerr_dH,
           label=r"$\Delta H$", error_kw=ec, zorder=3, **s_dH)
    ax.bar(x + offset, sub["mTdS_kJmol"], width, yerr=yerr_TdS,
           label=r"$-T\Delta S$", error_kw=ec, zorder=3, **s_TdS)
    ax.axhline(0, color="black", lw=0.6, zorder=2)
    if show_legend:
        ax.legend(fontsize=base * 0.6, frameon=False, ncol=1,
                  loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _style_thermo_axis(ax, sub, x, base, show_xlabels):
    ax.tick_params(axis="y", labelsize=base * 0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(True, axis="y", which="major", linestyle="--", alpha=0.35)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.15)
    ax.set_xticks(x)
    if show_xlabels:
        ax.set_xticklabels(_xtick_labels(sub), fontsize=base * 0.6)
    else:
        ax.tick_params(labelbottom=False, bottom=False)

    for i, is_kn in enumerate(sub["is_kn"]):
        if is_kn:
            ax.axvspan(i - 0.45, i + 0.45, color="#EEEEEE", alpha=0.6, zorder=0)


def plot_thermo_bars(df, output_path, N, method, cfg=None, title="",
                      palette=DEFAULT_PALETTE, alpha=DEFAULT_ALPHA,
                      show_errors=True):
    """Grouped bar chart of ΔG, ΔH, −TΔS per binding site (mean-fit values)."""
    cfg = cfg or PlotConfig()
    sub = df[(df["N"] == N) & (df["method"] == method)
             & (df["replicate"] == "mean")].copy()
    if len(sub) == 0:
        return None
    sub = sub.sort_values(["dataset", "site"])
    for col in ["dH_kJmol", "Cp_kJmolK", "dG_kJmol", "mTdS_kJmol",
                "e_dH", "e_dG", "e_mTdS"]:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    c_dG, c_dH, c_TdS = thermo_palette(palette)
    s_dG  = _bar_style(palette, "dG",  alpha, c_dG)
    s_dH  = _bar_style(palette, "dH",  alpha, c_dH)
    s_TdS = _bar_style(palette, "TdS", alpha, c_TdS)
    n = len(sub)
    x = np.arange(n)
    width = 0.22
    base = cfg.base_fontsize
    ec = {"capsize": 3, "elinewidth": 1.0, "markeredgewidth": 1.0}
    offset = width * 1.30  # slightly wider gaps between dG, dH, and -TdS
    vals = np.concatenate([
        sub["dG_kJmol"].to_numpy(dtype=float),
        sub["dH_kJmol"].to_numpy(dtype=float),
        sub["mTdS_kJmol"].to_numpy(dtype=float),
    ])
    if show_errors:
        errs = np.concatenate([
            sub["e_dG"].to_numpy(dtype=float),
            sub["e_dH"].to_numpy(dtype=float),
            sub["e_mTdS"].to_numpy(dtype=float),
        ])
    else:
        errs = np.zeros(3 * len(sub), dtype=float)
    segments = _bar_segments(vals, errs)
    fig_w, fig_h = _thermo_figsize(n)

    if len(segments) == 1:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _draw_thermo_bars_on_axis(
            ax, x, sub, width, offset, ec, s_dG, s_dH, s_TdS,
            show_legend=True, base=base, show_errors=show_errors,
        )
        ax.set_ylim(segments[0][0], segments[0][1])
        ax.set_ylabel("Energy (kJ/mol)", fontsize=base * 0.7)
        _style_thermo_axis(ax, sub, x, base, show_xlabels=True)
        fig.subplots_adjust(right=0.82, bottom=0.12, left=0.10, top=0.95)
        safe_savefig(fig, output_path, cfg.max_image_dim)
        plt.close(fig)
        return output_path

    ratios = _segment_height_ratios(segments)
    fig, axes = plt.subplots(
        len(segments), 1, sharex=True,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": ratios, "hspace": 0.07},
    )
    axes = np.atleast_1d(axes)
    for idx, (ax, (ymin, ymax, _kind)) in enumerate(zip(axes, segments)):
        _draw_thermo_bars_on_axis(
            ax, x, sub, width, offset, ec, s_dG, s_dH, s_TdS,
            show_legend=(idx == 0), base=base, show_errors=show_errors,
        )
        ax.set_ylim(ymin, ymax)
        _style_thermo_axis(ax, sub, x, base, show_xlabels=(idx == len(axes) - 1))
        if idx < len(axes) - 1:
            ax.spines["bottom"].set_visible(False)
        if idx > 0:
            ax.spines["top"].set_visible(False)

    for idx in range(len(axes) - 1):
        _draw_break_marks(axes[idx], axes[idx + 1], ratios[idx], ratios[idx + 1])

    axes[len(axes) // 2].set_ylabel("Energy (kJ/mol)", fontsize=base * 0.7)
    fig.subplots_adjust(right=0.82, bottom=0.12, left=0.10, top=0.95)
    safe_savefig(fig, output_path, cfg.max_image_dim)
    plt.close(fig)
    return output_path


# Make pandas importable here so plot_thermo_bars can use pd.to_numeric without
# forcing the top-level module to import pandas when only plot_vanthoff is used.
import pandas as pd
