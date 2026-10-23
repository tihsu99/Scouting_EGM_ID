"""Shared CMS presentation for the Matplotlib analysis plots."""
from pathlib import Path
from copy import deepcopy
import inspect
import math

import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, TextArea, HPacker, VPacker
from matplotlib.patches import Wedge, Circle
from matplotlib.text import Text


ID_COMPARISON_STYLE = {
    "formats": ["pdf", "png"],
    "wp_label_format": "{label}",
    "info_box": {"reserve_space": True, "show_details": False,
                 "heading": {"fontweight": "bold", "fontsize": 18}, "text": {"fontsize": 14},
                 "box": {"loc": "upper left", "pad": .5, "borderpad": .7, "frameon": False}},
    "efficiency_per_process": True, "efficiency_overlay_signal": True,
    "efficiency_linestyles": {"signal": "-", "background": "--"},
    "efficiency_ylim": [0, 1.2], "efficiency_x_errors": True,
    "wp_palette": ["#0072B2", "#56B4E9", "#3B4CC0", "#009E73", "#111111", "#707070"],
    "roc_box_aspect": 1, "roc_margins": {"x": .15, "y": .5},
    "max_series_per_panel": 24,
    "rc_params": {"figure.figsize": [12, 9], "font.size": 20, "legend.fontsize": 11,
                  "legend.title_fontsize": 12, "savefig.dpi": 150},
    "legend": {"loc": "upper center", "bbox_to_anchor": [.5, -.15], "ncol": 3,
               "frameon": False, "borderaxespad": 0},
    "cms_label": {"label": "Preliminary", "data": False, "loc": 0, "fontsize": 20},
    "cut_distributions": {
        "wp_palette": ["#8C1D18", "#C73E1D", "#F07C13", "#B8336A", "#A65F00", "#E4A11B"],
        "wp_lines": {"linestyle": "-", "linewidth": 2, "alpha": 1},
        "efficiency_curve": {"linestyle": "--", "linewidth": 2, "alpha": 1, "markersize": 4.5},
    },
    "composition": {"enabled": False, "size": 44, "fontsize": 9, "min_fraction_label": .1,
                    "box": {"loc": "lower right", "bbox_to_anchor": [0, .08, 1, 1],
                            "pad": 0, "borderpad": .7, "frameon": False}},
}


def resolve_plot_style(plot_style=None):
    """Share the ID comparison presentation while retaining explicit YAML overrides."""
    settings = deepcopy(plot_style or {})
    preset = settings.pop("preset", None)
    if preset is None:
        return settings
    if preset != "id_comparison":
        raise ValueError(f"Unknown plot style preset: {preset}")

    def merge(base, overrides):
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    return merge(deepcopy(ID_COMPARISON_STYLE), settings)


def configure_style(plot_style=None):
    settings = resolve_plot_style(plot_style)
    hep.style.use(hep.style.CMS)
    plt.rcParams.update({"axes.grid": False})
    plt.rcParams.update(settings.get("rc_params", {}))


def add_legend(ax, plot_style=None, **kwargs):
    options = dict(loc="lower right", frameon=False, borderaxespad=1.5)
    options.update(kwargs)
    options.update(resolve_plot_style(plot_style).get("legend", {}))
    return ax.legend(**options)


def series_panels(count, plot_style=None):
    limit = resolve_plot_style(plot_style).get("max_series_per_panel", 8)
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("max_series_per_panel must be a positive integer")
    panels = max(1, math.ceil(count / limit))
    columns = min(2, panels)
    rows = math.ceil(panels / columns)
    width, height = plt.rcParams["figure.figsize"]
    fig, grid = plt.subplots(rows, columns, squeeze=False, figsize=(columns * width, rows * height))
    axes = list(grid.flat)
    for ax in axes[panels:]:
        fig.delaxes(ax)
    return fig, axes[:panels], limit


def composition_panel(samples, settings):
    """Compact unweighted truth-composition pies."""
    size = settings.get("size", 34)
    fontsize = settings.get("fontsize", 8)
    pies = []
    for name, sample in samples.items():
        drawing = DrawingArea(size, size)
        start = 90.
        if not sample["rows"]:
            drawing.add_artist(Circle((size / 2, size / 2), size / 2, facecolor="none", edgecolor="0.6"))
        for category in sample["categories"]:
            fraction = category["fraction"]
            if not fraction:
                continue
            stop = start + 360 * fraction
            drawing.add_artist(Wedge((size / 2, size / 2), size / 2, start, stop,
                                     facecolor=category["color"], edgecolor="white", linewidth=.5))
            if fraction >= settings.get("min_fraction_label", .1):
                angle = math.radians((start + stop) / 2)
                drawing.add_artist(Text(size / 2 + .30 * size * math.cos(angle),
                                        size / 2 + .30 * size * math.sin(angle),
                                        f"{fraction:.0%}", ha="center", va="center", fontsize=fontsize - 1))
            start = stop
        pies.append(VPacker(children=[drawing, TextArea(name, textprops={"fontsize": fontsize})],
                            align="center", pad=0, sep=2))
    legend = [TextArea(category["label"], textprops={"fontsize": fontsize, "color": category["color"]})
              for category in settings["categories"].values()]
    return HPacker(children=[*pies, VPacker(children=legend, align="left", pad=0, sep=2)],
                   align="center", pad=0, sep=10)


def save_plot(fig, path, plot_style=None):
    settings = resolve_plot_style(plot_style)
    # Omit collision energy unless the campaign explicitly supplies it.
    label = dict(label="Preliminary", data=False, com=None, loc=0)
    label.update(settings.get("cms_label", {}))
    text = label.pop("label")
    # Use the requested status literally, independent of the MC/data flag.
    label.setdefault("llabel", text)
    # Older LCG mplhep accepts rlabel but has no com argument.
    if "com" not in inspect.signature(hep.label.exp_label).parameters:
        energy = label.pop("com")
        if label.get("rlabel") is None:
            right = []
            if label.get("lumi") is not None:
                right.append(label.get("lumi_format", "{0}").format(label["lumi"])
                             + r" $\mathrm{fb^{-1}}$")
            if label.get("year") is not None:
                right.append(str(label["year"]))
            if energy is not None:
                right.append(f"({energy} TeV)")
            label["rlabel"] = " ".join(right)
    # Finalize axes geometry before mplhep computes the header text offsets.
    fig.tight_layout()
    if settings.get("info_box", {}).get("reserve_space", False):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            for box in ax.artists:
                if not isinstance(box, AnchoredOffsetbox) or box.loc not in (1, 2, 9):
                    continue
                fraction = (box.get_window_extent(renderer).y0 - ax.bbox.y0 - 10) / ax.bbox.height
                if fraction <= 0:
                    raise ValueError("Information box does not fit inside the axes; reduce its size")
                low, high = ax.get_ylim()
                if ax.get_yscale() == "linear" and math.isfinite(ax.dataLim.y1):
                    ax.set_ylim(low, max(high, low + (ax.dataLim.y1 - low) / fraction))
    for ax in fig.axes:
        if ax.get_label() != "_secondary_axis":
            hep.cms.label(text, ax=ax, **label)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    formats = settings.get("formats")
    paths = [Path(path).with_suffix(f".{fmt}") for fmt in formats] if formats else [path]
    for output in paths:
        fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
