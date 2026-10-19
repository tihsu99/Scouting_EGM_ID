"""Shared CMS presentation for the Matplotlib analysis plots."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import mplhep as hep


def configure_style(plot_style=None):
    settings = plot_style or {}
    hep.style.use(hep.style.CMS)
    plt.rcParams.update({"axes.grid": False})
    plt.rcParams.update(settings.get("rc_params", {}))


def add_legend(ax, plot_style=None, **kwargs):
    options = dict(loc="lower right", frameon=False, borderaxespad=1.5)
    options.update(kwargs)
    options.update((plot_style or {}).get("legend", {}))
    return ax.legend(**options)


def series_panels(count, plot_style=None):
    limit = (plot_style or {}).get("max_series_per_panel", 8)
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


def save_plot(fig, path, plot_style=None):
    settings = plot_style or {}
    # Omit collision energy unless the campaign explicitly supplies it.
    label = dict(label="Preliminary", data=False, com=None, loc=0)
    label.update(settings.get("cms_label", {}))
    text = label.pop("label")
    # Use the requested status literally, independent of the MC/data flag.
    label.setdefault("llabel", text)
    for ax in fig.axes:
        hep.cms.label(text, ax=ax, **label)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
