#!/usr/bin/env python3
"""Plot DY reco/gen pT response in exclusive, nested correctionlib WP categories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import correctionlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import yaml

from apply_id import evaluate_id, id_evaluators, id_inputs
from inspect_energy_response import cache_nano, input_files, signal_frame, source_frames, validate_config
from workflow.plotting import configure_style, save_plot
import matplotlib.pyplot as plt


def check_same_candidates(original, completed, batch_size):
    left, right = pq.ParquetFile(original), pq.ParquetFile(completed)
    if left.metadata.num_rows != right.metadata.num_rows:
        raise ValueError("Cached Nano and original Parquet have different candidate counts")
    columns = left.schema_arrow.names
    for first, second in zip(left.iter_batches(batch_size=batch_size),
                             right.iter_batches(batch_size=batch_size, columns=columns)):
        if not first.to_pandas().equals(second.to_pandas()):
            raise ValueError("Cached Nano does not reproduce the original Parquet; refusing to attach WP inputs")


def complete_wp_inputs(path, required, config):
    """Complete old diagnostic caches from their retained Nano, without rerunning CMSSW."""
    path = Path(path)
    missing = set(required) - set(pq.ParquetFile(path).schema_arrow.names)
    if not missing:
        return str(path)
    receipt_path = Path(str(path) + ".json")
    if not receipt_path.is_file():
        raise ValueError(f"{path} lacks WP inputs {sorted(missing)} and has no diagnostic cache receipt")
    receipt = json.loads(receipt_path.read_text())
    if (not receipt.get("diagnostic_cache") or len(receipt.get("files_read", [])) != 1
            or receipt.get("gen_matching_relative_pt") is not None):
        raise ValueError("Automatic completion requires an unfiltered, single-Nano diagnostic cache")
    completed = path.with_name(path.stem + "_wp_inputs.parquet")
    if completed.exists():
        check_same_candidates(path, completed, config["batch_size"])
        if set(required) - set(pq.ParquetFile(completed).schema_arrow.names):
            raise ValueError(f"{completed} still lacks required WP inputs")
        return str(completed)
    nano = Path(receipt["files_read"][0]["path"])
    if not nano.is_file():
        nano = path.parent / "scouting_nano.root"
    if not nano.is_file():
        raise ValueError(f"{path} lacks WP inputs {sorted(missing)}; restore its scouting_nano.root or rerun the small MiniAOD diagnostic")
    if not isinstance(receipt.get("events_read"), int) or receipt["events_read"] <= 0:
        raise ValueError("Diagnostic cache receipt must record a positive event count")
    cached_config = dict(config, max_events=receipt["events_read"],
                         gen_matching_dR=receipt["gen_matching_dR"],
                         electron_response_max_relative_pt=receipt.get("electron_response_max_relative_pt"))
    print(f"Completing WP inputs from {nano}; original Parquet is retained", flush=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    metadata = Path(str(temporary) + ".json")
    try:
        cache_nano(cached_config, nano, temporary)
        if set(required) - set(pq.ParquetFile(temporary).schema_arrow.names):
            raise ValueError("Add the missing raw WP inputs to cache_columns in the diagnostic YAML")
        check_same_candidates(path, temporary, config["batch_size"])
        os.replace(temporary, completed)
        os.replace(metadata, str(completed) + ".json")
    finally:
        temporary.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
    print("WP-ready Parquet:", completed, flush=True)
    return str(completed)


def exclusive_categories(passed, names):
    """Fail if nesting is violated; do not silently modify exported WP decisions."""
    for tight, loose, tight_name, loose_name in zip(passed[:-1], passed[1:], names[:-1], names[1:]):
        violations = int(np.count_nonzero(tight & ~loose))
        if violations:
            raise ValueError(f"WPs are not nested: {violations} candidates pass {tight_name} but fail {loose_name}")
    categories = [passed[0]] + [loose & ~tight for tight, loose in zip(passed[:-1], passed[1:])]
    categories.append(~passed[-1])
    return np.argmax(np.stack(categories), axis=0)


def collect(config, files, corrections):
    pt_edges, _ = validate_config(config)
    settings = config["wp_response"]
    names = [item["name"] for item in settings["wps"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Provide unique WP names, ordered from strictest to loosest")
    evaluators = id_evaluators(corrections)
    if set(names) - set(evaluators):
        raise ValueError(f"Missing WPs {sorted(set(names) - set(evaluators))}; available: {list(evaluators)}")
    required = sorted({item.name for name in names for item in id_inputs(evaluators[name])})
    low, high, count = settings["binning"]
    if not np.isfinite([low, high]).all() or high <= low or type(count) is not int or count < 1:
        raise ValueError("wp_response.binning requires [min, max, positive integer bins]")
    if not settings["yscales"] or set(settings["yscales"]) - {"linear", "log"}:
        raise ValueError("wp_response.yscales must select linear and/or log")
    if not settings["groups"] or set(settings["groups"]) - set(config["hoe_groups"]):
        raise ValueError("Unknown or empty H/E groups for WP response")
    bins = np.linspace(low, high, count + 1)
    selected_files = files[:config["max_files"]] if config["max_files"] != -1 else files
    selected_files = [complete_wp_inputs(path, required, config) for path in selected_files]
    slices = [(f"pt{i}", a, b) for i, (a, b) in enumerate(zip(pt_edges[:-1], pt_edges[1:]))]
    if settings.get("inclusive", True):
        slices.insert(0, ("inclusive", -np.inf, np.inf))
    panels = []
    for region in config["region_definition"]:
        for key, low, high in slices:
            for group in settings["groups"]:
                panels.append(dict(region=region, slice=key, low=low, high=high, group=group,
                    sumw=np.zeros((len(names) + 1, count)), counts=np.zeros((len(names) + 1, count), dtype=np.int64),
                    entries=np.zeros(len(names) + 1, dtype=np.int64), total_sumw=np.zeros(len(names) + 1),
                    underflow_sumw=np.zeros(len(names) + 1), overflow_sumw=np.zeros(len(names) + 1)))
    report = dict(selected_signal=0, invalid_signal=0, original_inputs=files, nesting_verified=True)
    for raw in source_frames(dict(config, input_format="parquet"), selected_files, report, required):
        frame, invalid = signal_frame(raw, config)
        report["invalid_signal"] += invalid
        report["selected_signal"] += len(frame)
        if frame.empty:
            continue
        frame["component"] = exclusive_categories([evaluate_id(corrections, name, frame) for name in names], names)
        frame["pt_response"] = frame.pt / frame.gen_pt
        for panel in panels:
            subset = frame.query(config["region_definition"][panel["region"]]["requirement"])
            subset = subset.query(config["hoe_groups"][panel["group"]]["requirement"])
            subset = subset[(subset[config["pt_variable"]] >= panel["low"]) & (subset[config["pt_variable"]] < panel["high"])]
            for index in range(len(names) + 1):
                part = subset[subset.component == index]
                values, weights = part.pt_response.to_numpy(), part.weight.to_numpy()
                panel["entries"][index] += len(part)
                panel["total_sumw"][index] += weights.sum()
                panel["underflow_sumw"][index] += weights[values < bins[0]].sum()
                panel["overflow_sumw"][index] += weights[values > bins[-1]].sum()
                panel["sumw"][index] += np.histogram(values, bins, weights=weights)[0]
                panel["counts"][index] += np.histogram(values, bins)[0]
    return bins, panels, report


def component_styles(settings):
    wps = settings["wps"]
    labels = [wps[0]["label"]] + [f"{loose['label']} but not {tight['label']}" for tight, loose in zip(wps[:-1], wps[1:])]
    return labels + ["Fail " + wps[-1]["label"]], [item["color"] for item in wps] + [settings["fail_color"]]


def plot(config, bins, panels, output):
    configure_style(config.get("plot_style"))
    settings = config["wp_response"]
    labels, colors = component_styles(settings)
    for panel in tqdm(panels, desc="WP response", unit="panel", dynamic_ncols=True):
        for scale in settings["yscales"]:
            fig, ax = plt.subplots()
            positive, negative = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
            for values, label, color in zip(panel["sumw"], labels, colors):
                plus, minus = np.maximum(values, 0), np.minimum(values, 0)
                ax.fill_between(bins, np.r_[positive, positive[-1]], np.r_[positive + plus, (positive + plus)[-1]],
                                step="post", color=color, label=label, linewidth=0)
                ax.fill_between(bins, np.r_[negative, negative[-1]], np.r_[negative + minus, (negative + minus)[-1]],
                                step="post", color=color, linewidth=0)
                positive += plus
                negative += minus
            total = panel["sumw"].sum(axis=0)
            ax.step(bins, np.r_[total, total[-1]], where="post", color="black", linewidth=1.5, label="Total DY signal")
            note = ""
            if scale == "log":
                if (panel["sumw"] < 0).any():
                    ax.set_yscale("symlog", linthresh=max(np.abs(panel["sumw"]).max() * 1e-3, 1e-12))
                    note = "Signed bins: symmetric log scale"
                elif (positive > 0).any():
                    ax.set_yscale("log")
                    ax.set_ylim(bottom=positive[positive > 0].min() * .4, top=positive.max() * 12)
                else:
                    ax.set_yscale("log")
                    ax.set_ylim(.1, 1)
            else:
                ax.set_ylim(bottom=min(0., negative.min() * 1.15), top=max(positive.max() * 1.4, 1.))
            for reference in settings.get("reference_lines", []):
                ax.axvline(reference, color="0.45", linewidth=.8, linestyle="--")
            ax.set(xlabel=r"$p_T^{reco}/p_T^{gen}$", ylabel="Sum of generator weights" if config.get("weight_branch") else "Candidates",
                   xlim=(bins[0], bins[-1]))
            symbol = r"p_T^{gen}" if config["pt_variable"] == "gen_pt" else r"p_T^{reco}"
            pt_label = "Inclusive" if panel["slice"] == "inclusive" else (
                rf"${symbol} \geq {panel['low']:g}$ GeV" if np.isinf(panel["high"]) else
                rf"${panel['low']:g} \leq {symbol} < {panel['high']:g}$ GeV")
            text = (config["region_definition"][panel["region"]]["label"] + "\n" + pt_label + "\n" +
                    config["hoe_groups"][panel["group"]]["label"])
            ax.text(.03, .97, text, transform=ax.transAxes, va="top", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9))
            ax.legend(loc="upper right", fontsize=9)
            if not panel["entries"].sum():
                ax.text(.5, .45, "No DY signal candidates", transform=ax.transAxes, ha="center")
            fig.text(.12, -.01, f"Exclusive WP categories; raw-input correctionlib; {config.get('reco_requirement', 'all reco')}. "
                     f"N={panel['entries'].sum():,}; shown={panel['counts'].sum():,}.\n"
                     f"No relative-pT matching cut. Out-of-range candidates remain in summary.csv. {note}", fontsize=8)
            save_plot(fig, output / panel["region"] / f"pt_response_{panel['slice']}_{panel['group']}_{scale}.png", config.get("plot_style"))


def run(config, files, correction_path, output):
    corrections = correctionlib.CorrectionSet.from_file(str(correction_path))
    bins, panels, report = collect(config, files, corrections)
    labels, _ = component_styles(config["wp_response"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel in panels:
        for index, label in enumerate(labels):
            rows.append(dict(region=panel["region"], pt_slice=panel["slice"], group=panel["group"], component=label,
                entries=int(panel["entries"][index]), sumw=panel["total_sumw"][index],
                shown_entries=int(panel["counts"][index].sum()), underflow_sumw=panel["underflow_sumw"][index],
                overflow_sumw=panel["overflow_sumw"][index]))
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    report.update(correctionlib=str(Path(correction_path).resolve()),
                  correctionlib_sha256=hashlib.sha256(Path(correction_path).read_bytes()).hexdigest())
    serialized = [{key: value.tolist() if isinstance(value, np.ndarray) else
                   None if key in ("low", "high") and not np.isfinite(value) else value
                   for key, value in panel.items()} for panel in panels]
    (output / "histograms.json").write_text(json.dumps(dict(bins=bins.tolist(), components=labels, panels=serialized, report=report), indent=2, allow_nan=False) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    plot(config, bins, panels, output)
    print(f"Selected {report['selected_signal']:,} DY signal candidates. Output: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--files", required=True, nargs="+", help="Parquet from the small DY diagnostic")
    parser.add_argument("--id-correctionlib", required=True)
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--max-rows", type=int, default=-1, help="Candidate cap; default: all cached candidates")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    config.update(input_format="parquet", max_rows=args.max_rows)
    run(config, input_files(args.files, "parquet"), args.id_correctionlib, args.outputdir)


if __name__ == "__main__":
    main()
