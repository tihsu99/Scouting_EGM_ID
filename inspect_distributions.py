#!/usr/bin/env python3
"""Overlay unweighted candidate shapes from merged Parquet samples."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
from threading import Event, Lock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import yaml

from workflow.data import edges
from workflow.plotting import add_legend, configure_style, save_plot


def row_limit(config):
    limit = config.get("max_rows_per_sample")
    if limit is None or limit == -1:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("max_rows_per_sample must be a positive integer, null, or -1")
    return limit


def referenced_columns(config):
    """Collect column tokens, including quoted names, for Parquet projection."""
    expressions = list(config.get("define_variable", {}).values())
    expressions.append(config.get("requirement") or "")
    expressions.extend(spec.get("requirement") or "" for spec in config["region_definition"].values())
    expressions.extend(spec.get("expression") or name for name, spec in config["inspect_variable"].items())
    return {quoted or name for quoted, name in
            re.findall(r"`([^`]+)`|([A-Za-z_]\w*)", " ".join(expressions))}


def collect_histograms(config, inputdir):
    """Process samples in parallel; merge private histograms in the main thread."""
    processes = config["process"]
    variables = config["inspect_variable"]
    regions = config["region_definition"]
    if not processes or not variables or not regions:
        raise ValueError("Configure at least one process, variable, and region")
    batch_size = config.get("batch_size", 100000)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    workers = config.get("workers", 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    limit = row_limit(config)
    columns = referenced_columns(config)
    show_progress = config.get("progress", True)
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

    jobs = [(process, sample, paths) for process, samples in files.items()
            for sample, paths in samples.items()]
    available = {}
    with tqdm(total=sum(len(paths) for _, _, paths in jobs), desc="Scanning Parquet",
              unit="file", disable=not show_progress, dynamic_ncols=True) as progress:
        for _, sample, paths in jobs:
            available[sample] = 0
            for path in paths:
                available[sample] += pq.read_metadata(path).num_rows
                progress.update()
    total = sum(min(rows, limit) if limit is not None else rows for rows in available.values())
    lock, stop = Lock(), Event()

    def collect_sample(process, sample, paths, progress):
        local = {region: {var: dict(counts=np.zeros(len(bins) - 1, dtype=np.int64),
                                   rows=0, underflow=0, overflow=0, nonfinite=0)
                          for var, bins in binning.items()} for region in regions}
        report = dict(process=process, files=[str(path.resolve()) for path in paths],
                      files_read=[], available_rows=available[sample], rows=0,
                      selected_rows=0, regions=dict.fromkeys(regions, 0))
        target = min(available[sample], limit) if limit is not None else available[sample]
        for path in paths:
            if report["rows"] >= target or stop.is_set():
                break
            with path.open("rb") as source:
                parquet = pq.ParquetFile(source)
                selected_columns = [name for name in parquet.schema_arrow.names if name in columns]
                report["files_read"].append(str(path.resolve()))
                # Each worker owns its I/O and histograms; avoid nested Arrow thread pools.
                for batch in parquet.iter_batches(batch_size=min(batch_size, target - report["rows"]),
                                                  columns=selected_columns, use_threads=False):
                    if stop.is_set():
                        break
                    batch = batch.slice(0, min(batch.num_rows, target - report["rows"]))
                    frame = batch.to_pandas(use_threads=False)
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
                            hist = local[region][var]
                            hist["rows"] += len(values)
                            hist["nonfinite"] += int((~finite).sum())
                            values = values[finite]
                            hist["underflow"] += int((values < bins[0]).sum())
                            hist["overflow"] += int((values > bins[-1]).sum())
                            hist["counts"] += np.histogram(values, bins=bins)[0]
                    with lock:
                        progress.update(batch.num_rows)
                    if report["rows"] >= target:
                        break
        report["unread_rows"] = available[sample] - report["rows"]
        report["truncated"] = report["unread_rows"] > 0
        return local, report

    samples_report = {}
    with tqdm(total=total, desc="Candidates", unit="row", unit_scale=True,
              disable=not show_progress, dynamic_ncols=True) as progress:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(collect_sample, process, sample, paths, progress): (process, sample)
                       for process, sample, paths in jobs}
            try:
                for future in as_completed(pending):
                    process, sample = pending[future]
                    local, samples_report[sample] = future.result()
                    for region in regions:
                        for var in variables:
                            for key, value in local[region][var].items():
                                histograms[region][var][process][key] += value
                    with lock:
                        progress.set_postfix(samples=f"{len(samples_report)}/{len(jobs)}", refresh=False)
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
    samples_report = {sample: samples_report[sample] for _, sample, _ in jobs}
    return binning, histograms, samples_report


def density(hist, bins):
    counts = np.asarray(hist["counts"], dtype=float)
    total = counts.sum()
    return counts / (total * np.diff(bins)) if total else np.zeros_like(counts)


def plot_histograms(config, binning, histograms, outputdir):
    style = config.get("plot_style", {})
    configure_style(style)
    formats = config.get("formats", ["pdf", "png"])
    jobs = [(region, var, scale) for region, variables in histograms.items() for var in variables
            for scale in config["inspect_variable"][var].get(
                "yscales", config.get("yscales", ["linear", "log"]))]
    limit = row_limit(config)
    with tqdm(total=len(jobs) * len(formats), desc="Saving plots", unit="file",
              disable=not config.get("progress", True), dynamic_ncols=True) as progress:
        # Matplotlib stays on the main thread; both formats share one figure.
        for region, var, scale in jobs:
            region_spec = config["region_definition"][region]
            spec, bins = config["inspect_variable"][var], binning[var]
            fig, ax = plt.subplots()
            positive = []
            for process, hist in histograms[region][var].items():
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
            note = ("Unweighted candidates; each curve normalized within the shown range.\n"
                    "Sample bins combined by row count; no cross-section weighting.")
            if limit is not None:
                note += f"\nInput limited to first {limit:,} candidate rows/sample before cuts (not random)."
            ax.annotate(note, xy=(0, 0), xycoords="axes fraction", xytext=(0, -60),
                        textcoords="offset points", ha="left", va="top", fontsize=10)
            for index, extension in enumerate(formats):
                path = Path(outputdir) / region / f"{var}_{scale}.{extension}"
                if index == 0:
                    save_plot(fig, path, style)
                else:
                    fig.savefig(path, bbox_inches="tight")
                progress.update()


def run(config_path, inputdir, outputdir, workers=None, max_rows_per_sample=None, progress=None):
    config = yaml.safe_load(Path(config_path).read_text())
    for name, value in dict(workers=workers, max_rows_per_sample=max_rows_per_sample,
                            progress=progress).items():
        if value is not None:
            config[name] = value
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
    if config_output.exists():
        previous = yaml.safe_load(config_output.read_text())
        comparable = []
        for settings in (previous["config"], config):
            selection = {key: value for key, value in settings.items()
                         if key not in ("workers", "progress", "batch_size", "max_rows_per_sample")}
            selection["max_rows_per_sample"] = row_limit(settings)
            comparable.append(selection)
        if previous["inputdir"] != resolved["inputdir"] or comparable[0] != comparable[1]:
            raise ValueError("Configuration changed; use a new output directory to avoid mixing plots")
    binning, histograms, samples = collect_histograms(config, inputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    config_output.write_text(yaml.safe_dump(resolved, sort_keys=False))
    report = dict(weighting="one per candidate; not a cross-section-weighted prediction",
                  normalization="per-process unit area within plotted finite range",
                  sampling=dict(max_rows_per_sample=row_limit(config),
                                method="first rows in sorted file order, before selection; not random"),
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
    parser.add_argument("--workers", type=int, help="Concurrent sample-reading threads (overrides YAML)")
    parser.add_argument("--max-rows-per-sample", type=int,
                        help="Cap input candidate rows per sample before cuts; -1 reads all (overrides YAML)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = parser.parse_args()
    run(args.config, args.inputdir, args.outputdir, workers=args.workers,
        max_rows_per_sample=args.max_rows_per_sample, progress=False if args.no_progress else None)


if __name__ == "__main__":
    main()
