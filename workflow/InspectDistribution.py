import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from correctionlib import highlevel
from tqdm import tqdm

from workflow.data import edges, event_split, load_samples, reweight_training, training_weights
from workflow.plotting import add_legend, configure_style, save_plot


def apply_corrections(df, corrfile="corrections_all.json", progress=False):
    cs = highlevel.CorrectionSet.from_file(corrfile)
    for name in tqdm(cs.keys(), desc="EA corrections", unit="column",
                     dynamic_ncols=True, disable=not progress):
        df[name] = cs[name].evaluate(np.abs(df.eta.to_numpy(dtype=float)))
    return df


def InspectDistributionAnalysis(**kwargs):
    plot_style = kwargs.get("plot_style")
    progress = kwargs.get("progress", False)
    configure_style(plot_style)
    frames, normalization = load_samples(kwargs["inputdir"], kwargs["process"],
                                         kwargs.get("n_files", -1), kwargs.get("normalization"),
                                         max_rows_per_sample=kwargs.get("max_rows_per_sample"),
                                         batch_size=kwargs.get("batch_size", 100000), progress=progress)
    storedir, plotdir = Path(kwargs["storedir"]), Path(kwargs["plotdir"])
    storedir.mkdir(parents=True, exist_ok=True)
    (storedir / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    for category, frame in frames.items():
        if progress:
            tqdm.write(f"Preparing {category}: {len(frame):,} candidates after process selection")
        if kwargs.get("correction_lib"):
            apply_corrections(frame, kwargs["correction_lib"], progress=progress)
        for name, expression in tqdm(kwargs.get("define_variable", {}).items(),
                                     desc=f"{category}: features", unit="column",
                                     dynamic_ncols=True, disable=not progress):
            frame[name] = frame.eval(expression)
    for region in kwargs["regions"]:
        if progress:
            tqdm.write(f"{region}: selecting, splitting and reweighting candidates")
        requirement = kwargs["region_definition"][region]["requirement"]
        selected = {name: df.query(requirement).copy() for name, df in frames.items()}
        if any(df.empty for df in selected.values()):
            raise ValueError(f"Empty signal/background in {region}")
        region_plot = plotdir / region
        region_plot.mkdir(parents=True, exist_ok=True)
        train, valid = {}, {}
        for name, frame in selected.items():
            if kwargs.get("validation_only", False):
                valid[name] = frame
            else:
                train[name], valid[name] = event_split(frame, kwargs.get("train_ratio", 0.8),
                                                      kwargs.get("random_seed", 42), kwargs.get("split_columns"))
                if train[name].empty or valid[name].empty:
                    raise ValueError(f"Empty train/validation partition for {name} in {region}")
                train[name]["training_weight"] = training_weights(train[name], kwargs.get("training_weight_mode", "nonnegative"))
        if train:
            reweight_training(train["signal"], train["background"], kwargs.get("kinematic_reweighting"),
                              kwargs.get("reweight_binning", {"pt": [0, 200, 200], "eta": [-3, 3, 50]}), region_plot)
            for name, frame in train.items():
                if frame.training_weight.sum() <= 0:
                    raise ValueError(f"No positive training weight in {region}/{name}")
        outputs = [(split, name, frame) for split, datasets in (("train", train), ("valid", valid))
                   for name, frame in datasets.items()]
        for split, name, frame in tqdm(outputs, desc=f"{region}: saving tables", unit="file",
                                      dynamic_ncols=True, disable=not progress):
            path = storedir / region / split
            path.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path / f"{name}.parquet", index=False)
        # Plot training shapes after reweighting, or physical validation shapes.
        plotted = train or valid
        weight_col = "training_weight" if train else "weight"
        for var, spec in tqdm(kwargs.get("inspect_variable", {}).items(),
                             desc=f"{region}: plotting", unit="plot",
                             dynamic_ncols=True, disable=not progress):
            bins = edges(spec)
            fig, ax = plt.subplots()
            for name, frame in plotted.items():
                values = frame[var].to_numpy()
                hist = np.histogram(values, bins=bins, weights=frame[weight_col])[0]
                if hist.sum() > 0:
                    hist = hist / (hist.sum() * np.diff(bins))
                ax.stairs(hist, bins, label=name)
            ax.set(xlabel=var, ylabel="Density")
            add_legend(ax, plot_style, title=region)
            save_plot(fig, region_plot / f"{var}_density.pdf", plot_style)
        # Preserve the physical per-sample contributions to audit QCD stitching.
        if kwargs.get("stitch_binning"):
            bins = edges(kwargs["stitch_binning"])
            fig, ax = plt.subplots()
            report = {"pt_edges": bins.tolist(), "samples": {}}
            total = np.zeros(len(bins) - 1)
            for sample, frame in selected["background"].groupby("sample"):
                hist = np.histogram(frame.pt, bins=bins, weights=frame.weight)[0]
                sumw2 = np.histogram(frame.pt, bins=bins, weights=frame.weight**2)[0]
                report["samples"][sample] = {"sumw": hist.tolist(), "sumw2": sumw2.tolist()}
                total += hist
                ax.stairs(hist / np.diff(bins), bins, label=sample, alpha=0.65)
            ax.stairs(total / np.diff(bins), bins, color="black", linewidth=2, label="Total background")
            ax.set(xlabel=r"Electron $p_T$ [GeV]", ylabel="Weighted candidates / GeV", yscale="log")
            add_legend(ax, plot_style, title=region, ncol=2)
            save_plot(fig, region_plot / "background_stitching.pdf", plot_style)
            report["total_sumw"] = total.tolist()
            (region_plot / "background_stitching.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"storedir": str(storedir)}
