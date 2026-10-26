#!/usr/bin/env python3
"""Refit configured isolation pT terms and compare calibrations on held-out events."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import correctionlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from workflow.data import edges, event_split, load_samples, training_weights
from workflow.DeriveEffectiveArea import binned_profile, fit_and_plot
from workflow.exported_ids import read_json
from workflow.InspectDistribution import apply_corrections, define_variables
from workflow.plotting import add_legend, configure_style, save_plot
from workflow.working_points import load_id_config


def applied_coefficient(name, values, definitions):
    frame = pd.DataFrame({name: np.atleast_1d(values)})
    if name in definitions:
        define_variables(frame, {name: definitions[name]})
    return frame[name].to_numpy()


def compare_profiles(current, candidate, cfg, base, output):
    bins, style = edges(cfg["pt_binning"]), cfg.get("plot_style")
    configure_style(style)
    profile_label = (cfg["profile"].replace("contour", "") + "% quantile"
                     if cfg["profile"].startswith("contour") else "Mean")
    rows = []
    for region, definition in base["region_definition"].items():
        selected = current.eval(definition["requirement"]).to_numpy(dtype=bool)
        for variable, options in cfg["observables"].items():
            fig, axes = plt.subplots(2, 1, sharex=True)
            for label, frame, marker, color in (("Current", current, "s", "#D55E00"),
                                                ("Zprime refit", candidate, "o", "#0072B2")):
                subset = frame.loc[selected]
                for axis, quantity, values in ((axes[0], "absolute", subset[variable]),
                                               (axes[1], "relative", subset[variable] / subset.eval(cfg["relative_denominator"]))):
                    profile = binned_profile(subset.pt, values, subset.calibration_weight,
                                             bins, cfg["profile"], cfg["min_entries_per_bin"])
                    axis.plot(profile.x, profile.value, marker=marker, color=color, label=label)
                    rows.append(profile.assign(region=region, variable=variable, quantity=quantity, calibration=label))
            axes[0].set_ylabel(options["label"])
            axes[1].set_ylabel(f"Isolation / {cfg['relative_denominator']}")
            axes[1].set_xlabel(r"Electron $p_T$ [GeV]")
            axes[1].set_xlim(bins[0], bins[-1])
            for axis in axes:
                axis.axhline(0, color="0.6", linewidth=.7)
            add_legend(axes[0], style, title=f"{cfg['dataset']} — {region}\nHeld-out {profile_label}")
            save_plot(fig, output / "comparison" / region / f"{variable}.pdf", style)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(output / "profiles.csv", index=False)
    return table


def run(config_path, inputdir, correction_lib, outputdir):
    config_path, correction_lib = Path(config_path).resolve(), Path(correction_lib).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    evaluation, base, _ = load_id_config(config_path.parent / cfg["base_config"])
    dataset = evaluation["datasets"][cfg["dataset"]]
    definitions = base["workflow"]["InspectDistribution"]["define_variable"]
    recipes = base["workflow"]["DeriveEffectiveArea"]["EffectiveArea"]
    payload = read_json(correction_lib)
    original = {item["name"]: item for item in payload["corrections"]}
    bins = edges(cfg["pt_binning"])
    for name in cfg["fits"]:
        if name not in original or name not in recipes:
            raise ValueError(f"Missing calibration coefficient or pipeline fit recipe: {name}")
        if recipes[name]["x"] != "pt" or recipes[name]["y_type"].lower() == "none":
            raise ValueError(f"{name}: this comparison requires a binned pT fit")
    output = Path(outputdir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved = dict(configuration=cfg, pipeline=base, dataset=dataset,
                    inputdir=str(Path(inputdir).resolve()), correction_lib=str(correction_lib),
                    correction_sha256=hashlib.sha256(correction_lib.read_bytes()).hexdigest())
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    preparation = dict(base["workflow"]["InspectDistribution"], **cfg.get("preparation", {}))
    frames, loading = load_samples(inputdir, {cfg["dataset"]: {
        "samples": dict.fromkeys(dataset["samples"], 1.), "requirement": dataset["requirement"]}},
        **{key: preparation[key] for key in ("n_files", "normalization", "max_rows_per_sample", "batch_size",
                                            "progress", "sampling", "electron_response_max_relative_pt") if key in preparation},
        columns=preparation.get("input_columns"))
    selected = frames[cfg["dataset"]]
    train, valid = event_split(selected, preparation.get("train_ratio", .8),
                               preparation.get("random_seed", 42), preparation.get("split_columns"))
    if train.empty or valid.empty:
        raise ValueError(f"Both training and held-out {cfg['dataset']} partitions must be populated")
    del frames, selected
    for frame in (train, valid):
        frame["calibration_weight"] = training_weights(frame, "nonnegative")
        apply_corrections(frame, str(correction_lib))
        define_variables(frame, definitions)
    results, fit_profiles = [], []
    candidate_payload = deepcopy(payload)
    for name in cfg["fits"]:
        spec = recipes[name]
        values, eta_bins = [], spec["eta_bins"]
        for low, high in zip(eta_bins[:-1], eta_bins[1:]):
            mask = ((train.eta.abs() >= low) & (train.eta.abs() < high)
                    & (train.pt >= bins[0]) & (train.pt < bins[-1]))
            if spec.get("requirement"):
                mask &= train.eval(spec["requirement"])
            subset = train.loc[mask]
            y = subset.eval(spec["y"]).to_numpy()
            old_term = binned_profile(subset.pt, subset[name], subset.calibration_weight, bins, "mean",
                                      cfg["min_entries_per_bin"])
            result = fit_and_plot((low, high, subset.pt.to_numpy(), y, "pt", spec["y"], str(output / "fits"),
                spec["fitted_func"], name, spec["y_type"], bins, spec["x_title"], spec["y_title"], spec["title"],
                subset.calibration_weight.to_numpy(), spec.get("coefficient", "A1")),
                cfg.get("plot_style"), cfg["min_entries_per_bin"],
                reference=dict(x=old_term.x, y=old_term.value * old_term.x,
                               label="Current applied pT subtraction (no intercept)"))
            values.append(result["value"])
            old = subset[name].to_numpy()
            new = float(applied_coefficient(name, result["value"], definitions)[0])
            constant = np.allclose(old, old[0], rtol=1e-12, atol=0)
            results.append(dict(coefficient=name, **result, current_min=float(old.min()), current_max=float(old.max()),
                                new_applied=new, scale_factor=new / old[0] if constant and old[0] != 0 else np.nan,
                                delta_if_constant=new - old[0] if constant else np.nan))
            profile = binned_profile(subset.pt, y, subset.calibration_weight, bins, spec["y_type"], cfg["min_entries_per_bin"])
            fit_profiles.append(profile.assign(coefficient=name, eta_min=low, eta_max=high))
        for item in candidate_payload["corrections"]:
            if item["name"] == name:
                item["data"] = dict(nodetype="binning", input="eta", edges=list(map(float, eta_bins)), content=values, flow="clamp")
    candidate_path = output / "corrections_zprime_pt.json"
    # Validate the complete candidate before writing; all non-refitted corrections stay unchanged.
    correctionlib.CorrectionSet.from_string(json.dumps(candidate_payload))
    candidate_path.write_text(json.dumps(candidate_payload, indent=2) + "\n")
    pd.DataFrame(results).to_csv(output / "coefficient_comparison.csv", index=False)
    pd.concat(fit_profiles, ignore_index=True).to_csv(output / "fit_profiles.csv", index=False)
    candidate = valid.copy()
    apply_corrections(candidate, str(candidate_path))
    define_variables(candidate, definitions)
    profiles = compare_profiles(valid, candidate, cfg, base, output)
    report = dict(loading=loading, train_rows=len(train), heldout_rows=len(valid),
                  fit_parameters="Unweighted fit to populated profile bins; errors are fit covariance errors",
                  below_minimum_profile_bins=int(profiles.value.isna().sum()),
                  candidate_calibration=str(candidate_path),
                  full_id_updated=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(results)[["coefficient", "eta_min", "eta_max", "current_min", "current_max", "new_applied", "scale_factor"]].to_string(index=False))
    print(f"Comparison written to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--inputdir", required=True)
    parser.add_argument("--correction-lib", required=True, help="Existing EA/pT calibration JSON, not the full ID export")
    parser.add_argument("--outputdir", required=True, help="New comparison directory")
    args = parser.parse_args()
    run(args.config, args.inputdir, args.correction_lib, args.outputdir)


if __name__ == "__main__":
    main()
