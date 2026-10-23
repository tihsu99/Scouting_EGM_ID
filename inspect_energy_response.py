#!/usr/bin/env python3
"""Quick DY MiniAOD-to-Parquet diagnostic: H/E versus electron energy response."""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from tqdm import tqdm
import uproot
import yaml

from truth_matching import match_electron_truth
from ntuplizer import discover_files, root_url
from workflow.plotting import configure_style, save_plot


RECO_FIELDS = ("pt", "eta", "phi", "m", "hOverE")
GEN_FIELDS = ("pt", "eta", "phi", "mass", "pdgId", "status", "statusFlags")


def dataset_files(dataset, config):
    if config["input_format"] != "miniaod" or not dataset.endswith("/MINIAODSIM"):
        raise ValueError("--dataset requires input_format=miniaod and a DY MINIAODSIM dataset")
    validate_config(config)
    production = Path(config["miniAOD_production_config"])
    settings = yaml.safe_load(production.read_text())
    redirector = root_url(settings["production"]["input_redirector"] + "/placeholder").removesuffix("/placeholder")
    sample = dict(das=dataset, das_instance=config.get("das_instance", "prod/global"))
    limit = None if config["max_files"] == -1 else config["max_files"]
    with tqdm(total=1, desc="DAS discovery", unit="query", dynamic_ncols=True) as progress:
        files = discover_files(sample, production.parent, max_files=limit)
        progress.set_postfix(files=len(files))
        progress.update(1)
    print(f"Selected {len(files)} file(s) from {dataset}", flush=True)
    return [redirector.rstrip("/") + "/" + name for name in files]


def input_files(values, extension="root"):
    files = []
    for value in values:
        if value.startswith(("root://", "https://", "http://")):
            matches = [value]
        elif Path(value).is_dir():
            matches = sorted(str(p.resolve()) for p in Path(value).glob("*." + extension))
        else:
            matches = sorted(str(Path(p).resolve()) for p in glob.glob(value))
        if not matches:
            raise FileNotFoundError(value)
        files.extend(matches)
    if not files or len(files) != len(set(files)):
        raise ValueError("Provide nonempty, nonduplicated input files from the same DY sample")
    return files


def candidate_frame(arrays, config):
    reco = ak.zip({key: arrays["ScoutingElectron_" + key] for key in RECO_FIELDS})
    gen = ak.zip({key: arrays["GenPart_" + key] for key in GEN_FIELDS})
    # The quick cache and full production share this exact matching implementation.
    matched = match_electron_truth(reco, gen, config["gen_matching_dR"])

    def flat(values):
        return ak.to_numpy(ak.flatten(ak.fill_none(values, np.nan)))

    columns = {name: flat(matched[name]) for name in (*RECO_FIELDS, "status", "gen_pt", "gen_energy")}
    for name, branch in config.get("cache_columns", {}).items():
        columns[name] = flat(ak.broadcast_arrays(reco.pt, arrays[branch])[1])
    if config.get("weight_branch"):
        weights = ak.broadcast_arrays(reco.pt, arrays[config["weight_branch"]])[1]
        columns[config["parquet_weight_column"]] = flat(weights)
    return pd.DataFrame(columns)


def signal_frame(frame, config):
    frame = frame[frame.status.isin(config["signal_status"])].copy()
    frame["weight"] = frame[config["parquet_weight_column"]] if config.get("weight_branch") else 1.
    frame["reco_energy"] = np.sqrt((frame.pt * np.cosh(frame.eta)) ** 2 + frame.m ** 2)
    finite = np.isfinite(frame[["pt", "eta", "hOverE", "reco_energy", "gen_pt", "gen_energy", "weight"]]).all(axis=1)
    finite &= (frame.gen_energy > 0) & (frame.gen_pt > 0)
    invalid = int((~finite).sum())
    frame = frame[finite].copy()
    frame["response"] = frame.reco_energy / frame.gen_energy
    frame["relative_pt"] = abs(frame.pt - frame.gen_pt) / frame.gen_pt
    if config.get("reco_requirement"):
        frame = frame.query(config["reco_requirement"])
    return frame, invalid


def response_frame(arrays, config):
    return signal_frame(candidate_frame(arrays, config), config)


def source_frames(config, files, report, extra_columns=()):
    parquet = config["input_format"] == "parquet"
    cap = config["max_rows"] if parquet else config["max_events"]
    counter = "rows_read" if parquet else "events_read"
    report.update(events_read=None if parquet else 0, rows_read=0, files_read=[], files_requested=files)
    columns = ["pt", "eta", "m", "hOverE", "status", "gen_pt", "gen_energy"]
    if config.get("weight_branch"):
        columns.append(config["parquet_weight_column"])
    columns = list(dict.fromkeys(columns + list(extra_columns)))
    branches = ["ScoutingElectron_" + key for key in RECO_FIELDS] + ["GenPart_" + key for key in GEN_FIELDS]
    if config.get("weight_branch"):
        branches.append(config["weight_branch"])
    branches = list(dict.fromkeys(branches + list(config.get("cache_columns", {}).values())))
    with tqdm(total=None if cap == -1 else cap, desc="DY Parquet" if parquet else "Nano -> Parquet",
              unit="row" if parquet else "event", unit_scale=True, dynamic_ncols=True) as progress:
        for number, filename in enumerate(files, 1):
            if cap != -1 and report[counter] >= cap:
                break
            progress.set_postfix(file=f"{number}/{len(files)}")
            if parquet:
                source = pq.ParquetFile(filename)
                missing = set(columns) - set(source.schema_arrow.names)
                if missing:
                    raise ValueError(f"{filename} missing Parquet columns {sorted(missing)}; regenerate with gen_pt/gen_energy stored")
                available = source.metadata.num_rows
                stop = available if cap == -1 else min(available, cap - report[counter])
                record = dict(path=filename, rows=0, available_rows=available)
                report["files_read"].append(record)
                for batch in source.iter_batches(batch_size=config["batch_size"], columns=columns):
                    batch = batch.slice(0, min(batch.num_rows, stop - record["rows"]))
                    if not batch.num_rows:
                        break
                    record["rows"] += batch.num_rows
                    report["rows_read"] += batch.num_rows
                    progress.update(batch.num_rows)
                    yield batch.to_pandas()
                    if record["rows"] >= stop:
                        break
            else:
                with uproot.open(filename) as source:
                    tree = source[config["tree"]]
                    missing = set(branches) - set(tree.keys())
                    if missing:
                        raise ValueError(f"{filename} missing Nano branches: {sorted(missing)}; use --input-format miniaod for EDM input")
                    stop = tree.num_entries if cap == -1 else min(tree.num_entries, cap - report[counter])
                    report["files_read"].append(dict(path=filename, events=stop, available_events=tree.num_entries))
                    for arrays in tree.iterate(expressions=branches, entry_stop=stop,
                                               step_size=config["batch_size"], library="ak"):
                        frame = candidate_frame(arrays, config)
                        report["events_read"] += len(arrays)
                        report["rows_read"] += len(frame)
                        progress.update(len(arrays))
                        yield frame
    report["cap_reached"] = cap != -1 and report[counter] >= cap


def cache_nano(config, nano, output):
    """Keep all candidate truth categories; plotting performs the signal selection."""
    report, writer = {}, None
    try:
        for frame in source_frames(dict(config, input_format="nano"), [str(nano)], report):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No events in the temporary Nano; no diagnostic Parquet was created")
    Path(str(output) + ".json").write_text(json.dumps(dict(report, diagnostic_cache=True,
        gen_matching_dR=config["gen_matching_dR"], gen_matching_relative_pt=None), indent=2) + "\n")
    return report


def miniaod_cache(config, files):
    """Run the existing scouting CMSSW recipe locally on the selected files."""
    production = Path(config["miniAOD_production_config"]).resolve()
    settings = yaml.safe_load(production.read_text())
    runtime = settings["cmssw"]
    def path(value):
        return (production.parent / os.path.expandvars(value)).resolve()
    area, cmsset, pset = (path(runtime[key]) for key in ("base", "cmsset_default", "pset"))
    if not (area / "src").is_dir() or not cmsset.is_file() or not pset.is_file():
        raise ValueError("Missing CMSSW area/setup/pset from " + str(production))
    scratch = Path(tempfile.mkdtemp(prefix="dy-response-", dir=config["scratch_dir"])).resolve()
    (scratch / "tmp").mkdir()
    shutil.copyfile(pset, scratch / "nano_cfg.py")
    job = dict(input_files=[name if ":" in name else "file:" + str(Path(name).resolve()) for name in files],
               max_events=config["max_events"], threads=1, is_mc=True, physics=settings["physics"])
    if config.get("dataset"):
        job["dataset"] = config["dataset"]
    (scratch / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    script = '''set -eo pipefail
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS
export PYTHONNOUSERSITE=1
export SCRAM_ARCH="$3"
source "$1"
cd "$2/src"
eval "$(scram runtime -sh)"
test "$CMSSW_VERSION" = "$4"
cd "$5"
exec cmsRun -j FrameworkJobReport.xml nano_cfg.py job=job.json
'''
    command = ["bash", "--noprofile", "--norc", "-c", script, "dy-miniaod", str(cmsset), str(area),
               runtime["scram_arch"], runtime["release"], str(scratch)]
    print(f"MiniAOD -> Nano -> Parquet in {scratch}; {len(files)} input file(s), maxEvents={job['max_events']}", flush=True)
    with (scratch / "cmsRun.log").open("w") as log, tqdm(
            total=None if job["max_events"] == -1 else job["max_events"], desc="cmsRun", unit="event", dynamic_ncols=True) as bar:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              errors="replace", env=dict(os.environ, TMPDIR=str(scratch / "tmp"))) as process:
            for line in process.stdout:
                log.write(line)
                tqdm.write(line.rstrip())
                match = re.search(r"Begin processing the (\d+)(?:st|nd|rd|th) record", line)
                if match:
                    bar.update(max(0, int(match[1]) - bar.n))
            code = process.wait()
        if code:
            raise RuntimeError(f"cmsRun failed with exit {code}; log retained in {scratch / 'cmsRun.log'}")
        with uproot.open(scratch / "scouting_nano.root") as source:
            bar.update(max(0, source[config["tree"]].num_entries - bar.n))
    parquet = scratch / "electrons.parquet"
    report = cache_nano(config, scratch / "scouting_nano.root", parquet)
    print("Diagnostic Parquet:", parquet, flush=True)
    return parquet, report


def validate_config(config):
    if config.get("pt_variable") not in ("pt", "gen_pt"):
        raise ValueError("pt_variable must be pt or gen_pt")
    if config["input_format"] not in ("miniaod", "nano", "parquet"):
        raise ValueError("input_format must be miniaod, nano, or parquet")
    for name in ("batch_size", "max_events", "max_rows", "max_files"):
        value = config[name]
        if type(value) is not int or (value < 1 and not (name != "batch_size" and value == -1)):
            raise ValueError(name + " must be positive (limits also accept -1)")
    for name in ("gen_matching_dR", "relative_pt_threshold"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name + " must be finite and positive")
    if not config["signal_status"] or set(config["signal_status"]) - {1, 2}:
        raise ValueError("This diagnostic requires DY signal status 1 and/or 2")
    # Never accept a hidden matching cut that would sculpt the response tails.
    if config.get("gen_matching_relative_pt") is not None:
        raise ValueError("Do not apply a relative-pT matching cut to this diagnostic")
    pt_edges = np.asarray([np.inf if v is None else v for v in config["pt_bins"]], dtype=float)
    if (len(pt_edges) < 2 or not np.isfinite(pt_edges[:-1]).all()
            or np.isnan(pt_edges[-1]) or np.any(np.diff(pt_edges) <= 0)):
        raise ValueError("pt_bins must increase; only the last edge may be null")
    low, high, count = config["response_binning"]
    if not np.isfinite([low, high]).all() or high <= low or type(count) is not int or count < 1:
        raise ValueError("response_binning requires [min, max, positive integer bins]")
    if not config["region_definition"] or not config["plot_groups"] or not config["hoe_groups"]:
        raise ValueError("Provide regions, H/E groups, and plot_groups")
    if set(config["plot_groups"]) - config["hoe_groups"].keys():
        raise ValueError("Unknown H/E group in plot_groups")
    hoe_bins(config)
    if config.get("color_scale") not in ("auto", "linear"):
        raise ValueError("color_scale must be auto or linear")
    return pt_edges, np.linspace(low, high, count + 1)


def hoe_bins(config):
    low, high, count = config["hoe_binning"]
    if not np.isfinite([low, high]).all() or high <= low or type(count) is not int or count < 1:
        raise ValueError("hoe_binning requires [min, max, positive integer bins]")
    return np.linspace(low, high, count + 1)


def collect(config, files):
    pt_edges, bins = validate_config(config)
    histograms = []
    for region in config["region_definition"]:
        for index in range(len(pt_edges) - 1):
            for group in config["hoe_groups"]:
                histograms.append(dict(region=region, pt_bin=index, group=group,
                    entries=0, sumw=0., sumw2=0., response_sumw=0.,
                    below_half_sumw=0., above_one_half_sumw=0., pt_fail_sumw=0.,
                    underflow_sumw=0., overflow_sumw=0.,
                    counts=np.zeros(len(bins) - 1, dtype=np.int64), sumw_bins=np.zeros(len(bins) - 1)))
    xbins = hoe_bins(config)
    for hist in histograms:
        if hist["group"] in config["plot_groups"]:
            hist["sumw_2d"] = np.zeros((len(xbins) - 1, len(bins) - 1))
            hist["counts_2d"] = np.zeros((len(xbins) - 1, len(bins) - 1), dtype=np.int64)
    report = dict(selected_signal=0, invalid_signal=0,
                  energy_definition="sqrt((pt*cosh(eta))**2 + mass**2), for reco and gen",
                  matching="stored truth labels" if config["input_format"] == "parquet" else
                           "nearest final-state gen electron within DeltaR; no relative-pT cut")
    for raw in source_frames(config, files, report):
        frame, invalid = signal_frame(raw, config)
        report["invalid_signal"] += invalid
        report["selected_signal"] += len(frame)
        for region, spec in config["region_definition"].items():
            regional = frame.query(spec["requirement"])
            for index, (low, high) in enumerate(zip(pt_edges[:-1], pt_edges[1:])):
                selected = regional[(regional[config["pt_variable"]] >= low)
                                    & (regional[config["pt_variable"]] < high)]
                for hist in (h for h in histograms if h["region"] == region and h["pt_bin"] == index):
                    subset = selected.query(config["hoe_groups"][hist["group"]]["requirement"])
                    values, weights = subset.response.to_numpy(), subset.weight.to_numpy()
                    hist["entries"] += len(values)
                    hist["sumw"] += float(weights.sum())
                    hist["sumw2"] += float(np.square(weights).sum())
                    hist["response_sumw"] += float((values * weights).sum())
                    for key, mask in (("below_half_sumw", values < .5),
                                      ("above_one_half_sumw", values > 1.5),
                                      ("pt_fail_sumw", subset.relative_pt.to_numpy() >= config["relative_pt_threshold"]),
                                      ("underflow_sumw", values < bins[0]),
                                      ("overflow_sumw", values > bins[-1])):
                        hist[key] += float(weights[mask].sum())
                    hist["counts"] += np.histogram(values, bins=bins)[0]
                    hist["sumw_bins"] += np.histogram(values, bins=bins, weights=weights)[0]
                    if hist["group"] in config["plot_groups"]:
                        hist["sumw_2d"] += np.histogram2d(subset.hOverE, values, bins=[xbins, bins], weights=weights)[0]
                        hist["counts_2d"] += np.histogram2d(subset.hOverE, values, bins=[xbins, bins])[0].astype(np.int64)
    return bins, histograms, report

def plot(config, bins, histograms, output):
    configure_style(config.get("plot_style"))
    xbins = hoe_bins(config)
    panels = [h for h in histograms if h["group"] in config["plot_groups"]]
    for hist in tqdm(panels, desc="2D plots", unit="plot", dynamic_ncols=True):
        region, index = hist["region"], hist["pt_bin"]
        low, high = config["pt_bins"][index:index + 2]
        symbol = r"p_T^{gen}" if config["pt_variable"] == "gen_pt" else r"p_T^{reco}"
        pt_label = rf"${symbol} \geq {low:g}$ GeV" if high is None else rf"${low:g} \leq {symbol} < {high:g}$ GeV"
        values = hist["sumw_2d"].T
        limit = max(float(np.max(np.abs(values))), 1e-12)
        if np.any(values < 0):
            norm = SymLogNorm(linthresh=limit * 1e-3, vmin=-limit, vmax=limit) if config["color_scale"] == "auto" else Normalize(-limit, limit)
            cmap, plotted = config["signed_colormap"], values
        else:
            positive = values[values > 0]
            norm = LogNorm(vmin=float(positive.min()), vmax=max(limit, float(positive.min()) * 1.01)) if len(positive) and config["color_scale"] == "auto" else Normalize(0, max(limit, 1))
            cmap, plotted = config["colormap"], np.ma.masked_equal(values, 0)
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(xbins, bins, plotted, norm=norm, cmap=cmap, shading="auto")
        colorbar = fig.colorbar(mesh, ax=ax, label="Sum of generator weights" if config.get("weight_branch") else "Candidates")
        colorbar.ax.set_label("_secondary_axis")
        for reference in config.get("hoe_reference_lines", []):
            ax.axvline(reference, color="#D55E00", linewidth=1.4)
        for reference in config.get("response_reference_lines", []):
            ax.axhline(reference, color="0.5", linewidth=.8)
        ax.set(xlabel=r"$H/E$", ylabel=r"$E^{reco}/E^{gen}$", xlim=(xbins[0], xbins[-1]), ylim=(bins[0], bins[-1]))
        shown = int(hist["counts_2d"].sum())
        text = (config["region_definition"][region]["label"] + "\n" + pt_label + "\n" +
                config["sample_label"] + f"; N={hist['entries']:,}; shown={shown:,}")
        ax.text(.03, .97, text, transform=ax.transAxes, va="top", fontsize=11,
                bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
        if not hist["entries"]:
            ax.text(.5, .5, "No DY signal candidates", transform=ax.transAxes, ha="center", fontsize=13)
        note = (f"Signal status {config['signal_status']}; {config.get('reco_requirement', 'all reco')}; no additional relative-pT cut\n"
                "Out-of-range candidates are counted in summary.csv; colors show weighted entries per 2D bin.")
        fig.text(.12, -.01, note, fontsize=8, va="top")
        save_plot(fig, output / region / f"response2d_pt{index}_{hist['group']}.png", config.get("plot_style"))


def run(config, files, output):
    validate_config(config)
    original_files = list(files)
    files = files[:config["max_files"]] if config["max_files"] != -1 else files
    cache_report = None
    if config["input_format"] == "miniaod":
        cache, cache_report = miniaod_cache(config, files)
        files = [str(cache)]
        config = dict(config, input_format="parquet", max_rows=-1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bins, histograms, report = collect(config, files)
    report["original_inputs"] = original_files
    if config.get("dataset"):
        report["dataset"] = config["dataset"]
    if cache_report is not None:
        report["miniAOD_conversion"] = cache_report
        report["diagnostic_parquet"] = files[0]
    rows = []
    for hist in histograms:
        row = {key: value for key, value in hist.items() if not isinstance(value, np.ndarray)}
        index, total = hist["pt_bin"], hist["sumw"]
        if "counts_2d" in hist:
            row["shown_2d_entries"] = int(hist["counts_2d"].sum())
            row["outside_2d_sumw"] = hist["sumw"] - float(hist["sumw_2d"].sum())
        row.update(pt_variable=config["pt_variable"], pt_min=config["pt_bins"][index], pt_max=config["pt_bins"][index + 1])
        for name, source in (("mean_response", "response_sumw"), ("fraction_response_below_half", "below_half_sumw"),
                             ("fraction_response_above_one_half", "above_one_half_sumw"),
                             ("fraction_failing_proposed_pt_cut", "pt_fail_sumw")):
            row[name] = hist[source] / total if total else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    payload = dict(bins=bins.tolist(), hoe_bins=hoe_bins(config).tolist(), histograms=[{key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in hist.items()} for hist in histograms], report=report)
    (output / "histograms.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    plot(config, bins, histograms, output)
    print(f"Read {report['rows_read']:,} candidate rows; selected {report['selected_signal']:,} signal electrons. Output: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--dataset", help="DY DAS MINIAODSIM dataset; discover up to max-files inputs")
    sources.add_argument("--files", nargs="+", help="Explicit files, directories/globs, or root:// URLs from one DY sample")
    sources.add_argument("--file-list", help="Text file with one input path/URL per line")
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--input-format", choices=("miniaod", "nano", "parquet"))
    parser.add_argument("--max-rows", type=int, help="Parquet candidate-row cap, not an event cap")
    parser.add_argument("--max-files", type=int, help="Maximum input files to discover/read; -1 selects all")
    parser.add_argument("--max-events", type=int, help="Override the total event cap; -1 reads all inputs")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    for key in ("input_format", "max_events", "max_rows", "max_files"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["miniAOD_production_config"] = str((Path(args.config).resolve().parent / config["miniAOD_production_config"]).resolve())
    if args.dataset:
        config["dataset"] = args.dataset
        files = dataset_files(args.dataset, config)
    else:
        values = args.files if args.files else [line.strip() for line in Path(args.file_list).read_text().splitlines()
                                               if line.strip() and not line.lstrip().startswith("#")]
        files = input_files(values, "parquet" if config["input_format"] == "parquet" else "root")
    run(config, files, args.outputdir)


if __name__ == "__main__":
    main()
