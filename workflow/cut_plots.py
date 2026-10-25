"""Pre-ID sample shapes, exact WP boundaries, and single-variable efficiencies."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from workflow.data import edges, efficiency_stats
from workflow.working_points import cut_bounds
from workflow.plotting import resolve_plot_style, wp_color_indices, wp_marker_style


def plot_cut_distribution(samples, variable, specification, wp_sources, region, plot_style=None):
    """Return a figure and auditable histogram/cut metrics, using the supplied base weights."""
    style = resolve_plot_style(plot_style)
    settings = style.get("cut_distributions", {})
    options = dict(specification, **specification.get("by_region", {}).get(region, {}))
    bins = edges(options["binning"])
    xscale, yscale = options.get("xscale", "linear"), options.get("yscale", "linear")
    if xscale == "log" and bins[0] <= 0:
        raise ValueError(f"Logarithmic {variable} bins must be positive")
    fig, ax = plt.subplots()
    right = ax.twinx()
    right.set_label("_secondary_axis")
    right.set(ylabel="Single-cut efficiency", ylim=(0, 1.08))
    right.tick_params(axis="y", labelsize="small")
    right.yaxis.label.set_size("medium")
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette = settings.get("wp_palette", style.get("wp_palette", cycle))
    cut_line_style = dict(linestyle=":", linewidth=1.2, alpha=.7)
    cut_line_style.update(settings.get("wp_lines", {}))
    markers = style.get("source_markers", {})
    report = dict(variable=variable, region=region, selection="before ID", edges=bins.tolist(),
                  normalization="unit area within displayed range",
                  efficiency_denominator="all selected candidates before ID, including values outside the histogram range",
                  samples={}, cuts=[])
    handles, colors, efficiency_points = [], {}, {}
    for index, (name, (values, weights)) in enumerate(samples.items()):
        values, weights = np.asarray(values), np.asarray(weights, dtype=float)
        if not np.isfinite(weights).all():
            raise ValueError(f"Nonfinite distribution weights for {name}")
        finite = np.isfinite(values)
        histogram = np.histogram(values[finite], bins, weights=weights[finite])[0]
        sumw2 = np.histogram(values[finite], bins, weights=weights[finite] ** 2)[0]
        total = float(histogram.sum())
        if histogram.any() and total <= 0:
            raise ValueError(f"Nonpositive histogram normalization for {name}/{variable}")
        density = histogram / (total * np.diff(bins)) if total > 0 else np.zeros(len(bins) - 1)
        if yscale == "log" and (density < 0).any():
            raise ValueError(f"Negative histogram bins for {name}/{variable}; use a linear y axis")
        sample_style = {"linewidth": 2, **settings.get("samples", {}).get(name, {})}
        color = sample_style.pop("color", cycle[index % len(cycle)])
        label = sample_style.pop("label", name)
        colors[name] = color
        ax.stairs(density, bins, color=color, label=label, **sample_style)
        handles.append(Line2D([], [], color=color, linewidth=sample_style["linewidth"],
                              linestyle=sample_style.get("linestyle", "-"), label=label))
        efficiency_points[name] = {}
        report["samples"][name] = dict(rows=len(values), nonfinite_rows=int((~finite).sum()),
            underflow_rows=int((values[finite] < bins[0]).sum()), overflow_rows=int((values[finite] > bins[-1]).sum()),
            sumw=float(weights.sum()), in_range_sumw=total, bin_sumw=histogram.tolist(),
            bin_sumw2=sumw2.tolist(), density=density.tolist())

    pairs = [(source, wp) for source, wps in wp_sources.items() for wp in wps]
    indices = wp_color_indices(wp_sources, style)
    for index, (source, wp) in enumerate(pairs):
        bounds = cut_bounds(wp, variable)
        if not bounds:
            continue
        color = palette[indices[source, wp.name] % len(palette)]
        marker = markers.get(source, "o")
        label = style.get("wp_label_format", "{source} {label}").format(
            source=source, name=wp.name, label=wp.meta.get("label", wp.name))
        description = " and ".join(f"{operator} {value:.6g}" for operator, value in bounds)
        outside = any(not bins[0] <= value <= bins[-1] for _, value in bounds)
        handles.append(Line2D([], [], color=color, marker=marker, markersize=4, **cut_line_style,
                              **wp_marker_style(source, color, style),
                              label=f"{label}: {description}" + (" (outside range)" if outside else "")))
        for _, value in bounds:
            if bins[0] <= value <= bins[-1]:
                ax.axvline(value, color=color, **cut_line_style)
        entry = dict(source=source, wp=wp.name, bounds=bounds, outside_range=outside, samples={})
        for name, (values, weights) in samples.items():
            values, weights = np.asarray(values), np.asarray(weights, dtype=float)
            passed = np.isfinite(values)
            for operator, value in bounds:
                passed &= {"<": np.less, "≤": np.less_equal, ">": np.greater, "≥": np.greater_equal}[operator](values, value)
            efficiency, error, neff = efficiency_stats(passed, weights)
            entry["samples"][name] = dict(efficiency=float(efficiency) if np.isfinite(efficiency) else None,
                error=float(error) if np.isfinite(error) else None,
                effective_entries=float(neff) if np.isfinite(neff) else None,
                pass_rows=int(passed.sum()), pass_sumw=float(weights[passed].sum()))
            if np.isfinite(efficiency):
                for _, value in bounds:
                    if bins[0] <= value <= bins[-1]:
                        efficiency_points[name][(value, efficiency)] = True
        report["cuts"].append(entry)

    efficiency_style = dict(alpha=.35, linewidth=1.1, linestyle="--", marker="o", markersize=3)
    efficiency_style.update(settings.get("efficiency_curve", {}))
    for name, points in efficiency_points.items():
        points = sorted(points)
        if not points:
            continue
        right.plot(*zip(*points), color=colors[name], **efficiency_style)
    if any(efficiency_points.values()):
        handles.append(Line2D([], [], color="0.5", **efficiency_style, label="Single-cut efficiency"))
    ax.set(xlim=(bins[0], bins[-1]), xlabel=options.get("label", variable), ylabel="Normalized density",
           xscale=xscale, yscale=yscale)
    ax.margins(y=.25 if yscale == "linear" else .35)
    if yscale == "linear" and ax.get_ylim()[0] >= 0:
        ax.set_ylim(bottom=0)
    legend = dict(loc="upper right", ncol=1, frameon=False,
                  fontsize=10, borderaxespad=0)
    legend.update(settings.get("legend", {}))
    ax.legend(handles=handles, **legend)
    return fig, ax, report
