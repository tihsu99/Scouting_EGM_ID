#!/usr/bin/env python3
"""Overlay unweighted candidate shapes from merged Parquet samples."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import yaml

from workflow.data import edges
from workflow.plotting import add_legend, configure_style, save_plot


def collect_histograms(config, inputdir):
    """Stream all candidates once; merge counts before normalizing each process."""
    processes = config["process"]
    variables = config["inspect_variable"]
    regions = config["region_definition"]
    if not processes or not variables or not regions:
        raise ValueError("Configure at least one process, variable, and region")
    batch_size = config.get("batch_size", 100000)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    binning = {name: edges(spec["binning"]) for name, spec in variables.items()}
    histograms = {
        region: {
            var: {
                process: dict(counts=np.zeros(len(bins) - 1, dtype=np.int64),
                              rows=0, underflow=0, overflow=0, nonfinite=0)
                for process in processes
            }
            for var, bins in binning.items()
        }
        for region in regions
    }
    files, seen = {}, set()
    for process, spec in processes.items():
        samples = spec["samples"]
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Provide a nonempty sample list for {process}")
        files[process] = {}
        for sample in samples:
            if sample in seen:
                raise ValueError(f"Sample listed more than once: {sample}")
            seen.add(sample)
            paths = sorted((Path(inputdir) / sample).glob(config.get("file_pattern", "merged.parquet")))
            if not paths:
                raise FileNotFoundError(f"No input Parquet for {process}/{sample} in {inputdir}")
            files[process][sample] = paths

    samples_report = {}
    for process, samples in files.items():
        for sample, paths in samples.items():
            report = dict(process=process, files=[str(path.resolve()) for path in paths],
                          rows=0, selected_rows=0, regions=dict.fromkeys(regions, 0))
            samples_report[sample] = report
            for path in paths:
                print(f"Reading {process}/{sample}: {path}", flush=True)
                with path.open("rb") as source:
                    parquet = pq.ParquetFile(source)
                    for batch in parquet.iter_batches(batch_size=batch_size):
                        frame = batch.to_pandas()
                        report["rows"] += len(frame)
                        for name, expression in config.get("define_variable", {}).items():
                            frame[name] = frame.eval(expression)
                        if config.get("requirement"):
                            frame = frame.query(config["requirement"])
                        report["selected_rows"] += len(frame)
                        for region, region_spec in regions.items():
                            requirement = region_spec.get("requirement")
                            selected = frame.query(requirement) if requirement else frame
                            report["regions"][region] += len(selected)
                            for var, spec in variables.items():
                                expression = spec.get("expression")
                                values = (selected.eval(expression) if expression else selected[var])
                                values = values.to_numpy(dtype=float)
                                finite = np.isfinite(values)
                                bins = binning[var]
                                hist = histograms[region][var][process]
                                hist["rows"] += len(values)
                                hist["nonfinite"] += int((~finite).sum())
                                hist["underflow"] += int((values[finite] < bins[0]).sum())
                                hist["overflow"] += int((values[finite] > bins[-1]).sum())
                                hist["counts"] += np.histogram(values[finite], bins=bins)[0]
    return binning, histograms, samples_report


def density(hist, bins):
    counts = np.asarray(hist["counts"], dtype=float)
    total = counts.sum()
    return counts / (total * np.diff(bins)) if total else np.zeros_like(counts)


def plot_histograms(config, binning, histograms, outputdir):
    style = config.get("plot_style", {})
    configure_style(style)
    for region, variables in histograms.items():
        region_spec = config["region_definition"][region]
        for var, processes in variables.items():
            spec, bins = config["inspect_variable"][var], binning[var]
            for scale in spec.get("yscales", config.get("yscales", ["linear", "log"])):
                for extension in config.get("formats", ["pdf", "png"]):
                    fig, ax = plt.subplots()
                    positive = []
                    for process, hist in processes.items():
                        series = config["process"][process]
                        values = density(hist, bins)
                        positive.extend(values[values > 0])
                        count = int(hist["counts"].sum())
                        fraction = count / hist["rows"] if hist["rows"] else 0.0
                        label = (f'{series.get("label", process)} '
                                 f'(N={hist["rows"]:,}; shown {fraction:.1%})')
                        linestyle = series.get("linestyle", "-")
                        if isinstance(linestyle, list):
                            linestyle = tuple(linestyle)
                        ax.stairs(np.where(values > 0, values, np.nan) if scale == "log" else values,
                                  bins, baseline=None, label=label, color=series.get("color"),
                                  linestyle=linestyle, linewidth=1.8)
                    ax.set(xlabel=spec.get("label", var), ylabel="Unit-area density",
                           xlim=(bins[0], bins[-1]), xscale=spec.get("xscale", "linear"), yscale=scale)
                    if scale == "linear":
                        ax.set_ylim(0, max(positive) * 1.8 if positive else 1)
                    else:
                        ax.set_ylim(min(positive) * 0.5 if positive else 1e-6,
                                    max(positive) * 15 if positive else 1)
                    if not positive:
                        ax.text(0.5, 0.4, "No candidates in plotted range",
                                transform=ax.transAxes, ha="center", fontsize=14)
                    if spec.get("tick_labels"):
                        ticks = spec["tick_labels"]
                        ax.set_xticks(list(ticks))
                        ax.set_xticklabels(list(ticks.values()), fontsize=12)
                    title = region_spec.get("label", region)
                    selection = config.get("selection_label", config.get("requirement", "All candidates"))
                    ax.text(0.02, 0.97, f"{title}\n{selection}", transform=ax.transAxes,
                            ha="left", va="top", fontsize=12)
                    add_legend(ax, style, title="Candidates after selection")
                    ax.annotate("Unweighted candidates; each curve normalized within the shown range.\n"
                                "Sample bins combined by row count; no cross-section weighting.",
                                xy=(0, 0), xycoords="axes fraction", xytext=(0, -60),
                                textcoords="offset points", ha="left", va="top", fontsize=10)
                    save_plot(fig, Path(outputdir) / region / f"{var}_{scale}.{extension}", style)


def run(config_path, inputdir, outputdir):
    config = yaml.safe_load(Path(config_path).read_text())
    for name, spec in config["inspect_variable"].items():
        scales = spec.get("yscales", config.get("yscales", ["linear", "log"]))
        if not scales or set(scales) - {"linear", "log"}:
            raise ValueError(f"Invalid yscales for {name}: {scales}")
        if spec.get("xscale", "linear") not in ("linear", "log"):
            raise ValueError(f"Invalid xscale for {name}")
        if spec.get("xscale") == "log" and edges(spec["binning"])[0] <= 0:
            raise ValueError(f"Logarithmic x axis needs positive bin edges: {name}")
    if not config.get("formats", ["pdf", "png"]) or set(config.get("formats", [])) - {"pdf", "png"}:
        raise ValueError("formats must contain pdf and/or png")
    outputdir = Path(outputdir)
    resolved = dict(config=config, inputdir=str(Path(inputdir).resolve()))
    config_output = outputdir / "resolved_config.yaml"
    if config_output.exists() and yaml.safe_load(config_output.read_text()) != resolved:
        raise ValueError("Configuration changed; use a new output directory to avoid mixing plots")
    binning, histograms, samples = collect_histograms(config, inputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    config_output.write_text(yaml.safe_dump(resolved, sort_keys=False))
    report = dict(weighting="one per candidate; not a cross-section-weighted prediction",
                  normalization="per-process unit area within plotted finite range",
                  samples=samples, histograms={})
    for region, variables in histograms.items():
        report["histograms"][region] = {}
        for var, processes in variables.items():
            report["histograms"][region][var] = dict(
                edges=binning[var].tolist(),
                processes={name: dict(hist, counts=hist["counts"].tolist(),
                                      density=density(hist, binning[var]).tolist())
                           for name, hist in processes.items()})
    (outputdir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    plot_histograms(config, binning, histograms, outputdir)
    print(f"Plots and summary written to {outputdir.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML with process groups, selections, variables and binning")
    parser.add_argument("--inputdir", required=True, help="Mounted merged directory containing sample folders")
    parser.add_argument("--outputdir", required=True)
    args = parser.parse_args()
    run(args.config, args.inputdir, args.outputdir)


if __name__ == "__main__":
    main()
