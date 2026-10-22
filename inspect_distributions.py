#!/usr/bin/env python3
"""Inspect candidate shapes or receipt-normalized spectra from merged Parquet."""
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

from workflow.data import edges, sample_normalizations
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
    columns = {quoted or name for quoted, name in
               re.findall(r"`([^`]+)`|([A-Za-z_]\w*)", " ".join(expressions))}
    if config.get("normalization"):
        columns.add(config["normalization"].get("weight_column", "Generator.weight"))
    return columns


def empty_histogram(bins, weighted):
    hist = dict(counts=np.zeros(len(bins) - 1, dtype=np.int64),
                rows=0, underflow=0, overflow=0, nonfinite=0)
    if weighted:
        hist.update(sumw=np.zeros(len(bins) - 1), sumw2=np.zeros(len(bins) - 1),
                    underflow_sumw=0., overflow_sumw=0., nonfinite_sumw=0.)
    return hist


def total_histogram(processes):
    return {key: sum(hist[key] for hist in processes.values())
            for key in next(iter(processes.values()))}


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
    normalization = config.get("normalization") or {}
    if normalization and limit is not None:
        raise ValueError("max_rows_per_sample cannot be combined with cross-section normalization; "
                         "use -1 to retain the receipt's full event denominator")
    weight_column = normalization.get("weight_column", "Generator.weight")
    columns = referenced_columns(config)
    show_progress = config.get("progress", True)
    binning = {name: edges(spec["binning"]) for name, spec in variables.items()}
    histograms = {
        region: {
            var: {
                process: empty_histogram(bins, normalization)
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
    normalized = sample_normalizations({sample: paths for _, sample, paths in jobs}, normalization) if normalization else {}
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
        local = {region: {var: empty_histogram(bins, normalization)
                          for var, bins in binning.items()} for region in regions}
        report = dict(process=process, files=[str(path.resolve()) for path in paths],
                      files_read=[], available_rows=available[sample], rows=0,
                      selected_rows=0, regions=dict.fromkeys(regions, 0))
        if normalization:
            report["normalization"] = normalized[sample]
        target = min(available[sample], limit) if limit is not None else available[sample]
        for path in paths:
            if report["rows"] >= target or stop.is_set():
                break
            with path.open("rb") as source:
                parquet = pq.ParquetFile(source)
                selected_columns = [name for name in parquet.schema_arrow.names if name in columns]
                if normalization and weight_column not in selected_columns:
                    raise ValueError(f"Missing generator weight column {weight_column}: {path}")
                report["files_read"].append(str(path.resolve()))
                # Each worker owns its I/O and histograms; avoid nested Arrow thread pools.
                for batch in parquet.iter_batches(batch_size=min(batch_size, target - report["rows"]),
                                                  columns=selected_columns, use_threads=False):
                    if stop.is_set():
                        break
                    batch = batch.slice(0, min(batch.num_rows, target - report["rows"]))
                    frame = batch.to_pandas(use_threads=False)
                    report["rows"] += len(frame)
                    if normalization:
                        weights = frame[weight_column].to_numpy(dtype=float) * normalized[sample]["normalization_factor"]
                        if not np.isfinite(weights).all():
                            raise ValueError(f"Non-finite weights in {sample}")
                    for name, expression in config.get("define_variable", {}).items():
                        frame[name] = frame.eval(expression)
                    if config.get("requirement"):
                        frame = frame.query(config["requirement"])
                    report["selected_rows"] += len(frame)
                    for region, region_spec in regions.items():
                        requirement = region_spec.get("requirement")
                        selected = frame.query(requirement) if requirement else frame
                        report["regions"][region] += len(selected)
                        if normalization:
                            weights = selected[weight_column].to_numpy(dtype=float) * normalized[sample]["normalization_factor"]
                        for var, spec in variables.items():
                            expression = spec.get("expression")
                            values = (selected.eval(expression) if expression else selected[var])
                            values = values.to_numpy(dtype=float)
                            finite = np.isfinite(values)
                            bins = binning[var]
                            hist = local[region][var]
                            hist["rows"] += len(values)
                            hist["nonfinite"] += int((~finite).sum())
                            if normalization:
                                hist["nonfinite_sumw"] += float(weights[~finite].sum())
                                hist["underflow_sumw"] += float(weights[finite & (values < bins[0])].sum())
                                hist["overflow_sumw"] += float(weights[finite & (values > bins[-1])].sum())
                                hist["sumw"] += np.histogram(values[finite], bins=bins, weights=weights[finite])[0]
                                hist["sumw2"] += np.histogram(values[finite], bins=bins, weights=np.square(weights[finite]))[0]
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


def density(hist, bins, mode="unit_area"):
    counts = np.asarray(hist.get("sumw", hist["counts"]), dtype=float)
    if mode == "yield_density":
        return counts / np.diff(bins)
    if mode != "unit_area":
        raise ValueError(f"Unknown histogram_normalization: {mode}")
    total = counts.sum()
    return counts / (total * np.diff(bins)) if total else np.zeros_like(counts)


def plot_kind(config):
    kind = config.get("plot_kind", "overlay")
    if kind not in ("overlay", "stacked"):
        raise ValueError(f"Unknown plot_kind: {kind}")
    if kind == "stacked" and config.get("histogram_normalization", "unit_area") != "yield_density":
        raise ValueError("Stacked contributions require histogram_normalization: yield_density")
    return kind


def plot_histograms(config, binning, histograms, outputdir):
    style = config.get("plot_style", {})
    configure_style(style)
    formats = config.get("formats", ["pdf", "png"])
    jobs = [(region, var, scale) for region, variables in histograms.items() for var in variables
            for scale in config["inspect_variable"][var].get(
                "yscales", config.get("yscales", ["linear", "log"]))]
    limit = row_limit(config)
    mode = config.get("histogram_normalization", "unit_area")
    stacked = plot_kind(config) == "stacked"
    weighted = bool(config.get("normalization"))
    with tqdm(total=len(jobs) * len(formats), desc="Saving plots", unit="file",
              disable=not config.get("progress", True), dynamic_ncols=True) as progress:
        # Matplotlib stays on the main thread; both formats share one figure.
        for region, var, scale in jobs:
            region_spec = config["region_definition"][region]
            spec, bins = config["inspect_variable"][var], binning[var]
            fig, ax = plt.subplots()
            positive, negative = [], []
            cumulative = np.zeros(len(bins) - 1)
            curves = [(name, config["process"][name], hist) for name, hist in histograms[region][var].items()]
            component_count = len(curves)
            if config.get("total"):
                curves.append(("Total", config["total"], total_histogram(histograms[region][var])))
            for index, (name, series, hist) in enumerate(curves):
                values = density(hist, bins, mode)
                positive.extend(values[values > 0])
                negative.extend(values[values < 0])
                count = int(hist["counts"].sum())
                fraction = count / hist["rows"] if hist["rows"] else 0.0
                label = (f'{series.get("label", name)} '
                         f'(N={hist["rows"]:,}; shown {fraction:.1%})')
                linestyle = series.get("linestyle", "-")
                if isinstance(linestyle, list):
                    linestyle = tuple(linestyle)
                if stacked and index < component_count:
                    if np.any(values < 0):
                        plt.close(fig)
                        raise ValueError("Stacked plots require nonnegative bin yields; use plot_kind: overlay for signed contributions")
                    baseline = cumulative.copy()
                    cumulative = baseline + values
                    ax.stairs(cumulative, bins, baseline=baseline, fill=True,
                              label=label, facecolor=series.get("color"), edgecolor="white", linewidth=0.3)
                else:
                    ax.stairs(np.where(values > 0, values, np.nan) if scale == "log" else values,
                              bins, baseline=None, label=label, color=series.get("color"),
                              linestyle=linestyle, linewidth=series.get("linewidth", 1.8))
            if stacked:
                positive = cumulative[cumulative > 0].tolist()
            ylabel = "Unit-area density"
            if mode == "yield_density":
                ylabel = "Weighted candidates / bin width" if weighted else "Candidates / bin width"
            ax.set(xlabel=spec.get("label", var), ylabel=spec.get("ylabel", ylabel),
                   xlim=(bins[0], bins[-1]), xscale=spec.get("xscale", "linear"), yscale=scale)
            if scale == "linear":
                ax.set_ylim(min(negative) * 1.2 if negative else 0, max(positive) * 1.8 if positive else 1)
            else:
                ax.set_ylim(min(positive) * 0.5 if positive else 1e-6,
                            max(positive) * 15 if positive else 1)
            if not positive and (scale == "log" or not negative):
                ax.text(0.5, 0.4, "No positive bins in plotted range" if weighted else "No candidates in plotted range",
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
            if weighted:
                lumi = config["normalization"].get("luminosity_pb", 1.0)
                note = (rf"Cross-section and generator weights; $L={lumi:g}$ pb$^{{-1}}$. "
                        "Denominators from preselection event receipts.\n")
                note += ("Each curve normalized within the shown range." if mode == "unit_area" else
                         "Bin yields divided by bin width; relative sample normalization preserved.")
                if scale == "log" and negative:
                    note += " Nonpositive bins omitted on log axes."
            else:
                note = ("Unweighted candidates; each curve normalized within the shown range.\n"
                        "Sample bins combined by row count; no cross-section weighting.")
                if mode == "yield_density":
                    note = "Unweighted candidate counts divided by bin width; no cross-section weighting."
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


def load_config(config_path):
    """Apply top-level overrides from one base YAML and resolve normalization at source."""
    path = Path(config_path).resolve()
    overrides = yaml.safe_load(path.read_text())
    base_path = path.parent / overrides.pop("base_config") if "base_config" in overrides else None
    config = yaml.safe_load(base_path.read_text()) if base_path else {}
    if "base_config" in config:
        raise ValueError("Inspection supports one base_config level")
    norm_parent = path.parent
    if base_path and "normalization_config" not in overrides:
        norm_parent = base_path.parent
    config.update(overrides)
    if "normalization_config" in config:
        if config.get("normalization"):
            raise ValueError("Use normalization or normalization_config, not both")
        source = (norm_parent / config.pop("normalization_config")).resolve()
        config["normalization"] = yaml.safe_load(source.read_text())["normalization"]
        if not config["normalization"]:
            raise ValueError(f"Missing normalization settings in {source}")
    return config


def run(config_path, inputdir, outputdir, workers=None, max_rows_per_sample=None, progress=None,
        plot_only=False):
    config = load_config(config_path)
    plot_kind(config)
    for name, value in dict(workers=workers, max_rows_per_sample=max_rows_per_sample,
                            progress=progress).items():
        if value is not None:
            config[name] = value
    mode = config.get("histogram_normalization", "unit_area")
    if mode not in ("unit_area", "yield_density"):
        raise ValueError(f"Unknown histogram_normalization: {mode}")
    if config.get("normalization") and row_limit(config) is not None:
        raise ValueError("max_rows_per_sample cannot be combined with cross-section normalization; use -1")
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
                         if key not in ("workers", "progress", "batch_size", "max_rows_per_sample", "plot_kind")}
            selection["max_rows_per_sample"] = row_limit(settings)
            comparable.append(selection)
        if previous["inputdir"] != resolved["inputdir"] or comparable[0] != comparable[1]:
            raise ValueError("Configuration changed; use a new output directory to avoid mixing plots")
    if plot_only:
        if not config_output.is_file():
            raise FileNotFoundError(f"Missing saved configuration: {config_output}")
        report = json.loads((outputdir / "summary.json").read_text())
        binning, histograms = {}, {}
        for region, variables in report["histograms"].items():
            histograms[region] = {}
            for var, saved in variables.items():
                binning[var] = np.asarray(saved["edges"], dtype=float)
                if not np.array_equal(binning[var], edges(config["inspect_variable"][var]["binning"])):
                    raise ValueError(f"Saved binning does not match configuration: {region}/{var}")
                histograms[region][var] = {
                    name: {key: np.asarray(value, dtype=np.int64 if key == "counts" else float)
                           if key in ("counts", "sumw", "sumw2") else value
                           for key, value in hist.items() if key != "density"}
                    for name, hist in saved["processes"].items()}
        print(f"Replotting saved histograms from {outputdir / 'summary.json'}; no Parquet reads", flush=True)
        plot_histograms(config, binning, histograms, outputdir)
        config_output.write_text(yaml.safe_dump(resolved, sort_keys=False))
        print(f"Plots written to {outputdir.resolve()}", flush=True)
        return report
    binning, histograms, samples = collect_histograms(config, inputdir)
    outputdir.mkdir(parents=True, exist_ok=True)
    config_output.write_text(yaml.safe_dump(resolved, sort_keys=False))
    report = dict(weighting="one per candidate; not a cross-section-weighted prediction",
                  normalization="per-process unit area within plotted finite range",
                  sampling=dict(max_rows_per_sample=row_limit(config),
                                method="first rows in sorted file order, before selection; not random"),
                  samples=samples, histograms={})
    if config.get("normalization"):
        weight_column = config["normalization"].get("weight_column", "Generator.weight")
        report["weighting"] = f"{weight_column} * cross_section_pb * luminosity_pb / preselection sum_genweight"
        report["sampling"]["method"] = "all input candidate rows; denominators from included event receipts"
        report["sumw2_definition"] = "sum of squared candidate weights; not an event-cluster variance"
    if mode == "yield_density":
        report["normalization"] = "bin yields divided by bin width; no per-curve unit-area scaling"

    def serialize_histogram(hist, bins):
        saved = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in hist.items()}
        saved["density"] = density(hist, bins, mode).tolist()
        return saved

    for region, variables in histograms.items():
        report["histograms"][region] = {}
        for var, processes in variables.items():
            report["histograms"][region][var] = dict(
                edges=binning[var].tolist(),
                processes={name: serialize_histogram(hist, binning[var])
                           for name, hist in processes.items()})
            if config.get("total"):
                report["histograms"][region][var]["total"] = serialize_histogram(total_histogram(processes), binning[var])
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
    parser.add_argument("--plot-only", action="store_true",
                        help="Replot saved summary.json without reading Parquet; use the original config and cap")
    args = parser.parse_args()
    run(args.config, args.inputdir, args.outputdir, workers=args.workers,
        max_rows_per_sample=args.max_rows_per_sample, progress=False if args.no_progress else None,
        plot_only=args.plot_only)


if __name__ == "__main__":
    main()
