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
from workflow.DeriveEffectiveArea import binned_profile, fit_and_plot, parse_fitted_function
from workflow.exported_ids import read_json
from workflow.InspectDistribution import apply_corrections, define_variables
from workflow.plotting import add_legend, configure_style, save_plot
from workflow.working_points import load_id_config


def applied_coefficient(name, values, definitions):
    frame = pd.DataFrame({name: np.atleast_1d(values)})
    if name in definitions:
        define_variables(frame, {name: definitions[name]})
    return frame[name].to_numpy()


def relocate_config_paths(value, config_dir):
    """Keep config-relative references valid in the emitted candidate ID recipe."""
    if isinstance(value, dict):
        return {key: relocate_config_paths(item, config_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_config_paths(item, config_dir) for item in value]
    return value.replace("{config_dir}", str(config_dir)) if isinstance(value, str) else value


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
            labels = cfg.get("comparison_labels", {})
            for label, frame, marker, color in ((labels.get("current", "Existing calibration"), current, "s", "#D55E00"),
                                                (labels.get("candidate", "Zprime refit"), candidate, "o", "#0072B2")):
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


def profile_diagnostics(train, valid, candidate, cfg, fits, output):
    """Compare profile shapes on alternative axes without exporting their fit coefficients."""
    profiles, results = [], []
    for name, options in cfg.get("profile_diagnostics", {}).items():
        spec = fits[name][0]
        observable = options["observable"]
        for axis, axis_options in options["axes"].items():
            bins = edges(axis_options.get("binning", cfg["pt_binning"]))
            fit_range = spec.get("fit_range") or [bins[0], bins[-1]]
            directory = output / "diagnostics" / name / axis
            for low, high in zip(spec["eta_bins"][:-1], spec["eta_bins"][1:]):
                masks = [((frame.eta.abs() >= low) & (frame.eta.abs() < high)
                          & (frame.eval(spec["requirement"]) if spec.get("requirement") else True))
                         for frame in (train, valid)]
                training, heldout = train.loc[masks[0]], valid.loc[masks[1]]
                updated = candidate.loc[masks[1]]
                measured = {}
                for stage, frame, expression, split in (
                        ("pileup_only_train", training, spec["y"], "train"),
                        ("pileup_only_heldout", heldout, spec["y"], "heldout"),
                        ("current_corrected", heldout, observable, "heldout"),
                        ("candidate_corrected", updated, observable, "heldout")):
                    profile = binned_profile(frame.eval(axis), frame.eval(expression), frame.calibration_weight,
                                             bins, spec["y_type"], cfg["min_entries_per_bin"])
                    profile["within_fit_range"] = profile.x.between(*fit_range)
                    measured[stage] = profile
                    profiles.append(profile.assign(coefficient=name, axis=axis, stage=stage, split=split,
                                                   expression=expression, eta_min=low, eta_max=high))

                # Native calibration application above still uses the configured reco inputs.
                style = dict(cfg.get("plot_style") or {})
                configure_style(style)
                fig, axes = plt.subplots(2, 1, sharex=True, figsize=options.get("figsize"))
                quantile = (spec["y_type"].replace("contour", "") + "th percentile"
                            if spec["y_type"].startswith("contour") else "Mean")
                for stage, label, marker, color in (("pileup_only_train", "Training", "o", "0.5"),
                                                   ("pileup_only_heldout", "Held-out", "s", "#0072B2")):
                    profile = measured[stage]
                    axes[0].plot(profile.x, profile.value, linestyle="none", marker=marker,
                                 color=color, fillstyle="none" if stage.endswith("train") else "full",
                                 label=f"{label}: {quantile}")
                for model, model_options in options["models"].items():
                    formula = model_options["function"]
                    function, parameters, _ = parse_fitted_function(formula)
                    profile_label = spec.get("fit_labels", {}).get("profile", f"{cfg['dataset']}: {quantile}")
                    model_style = dict(style, fit_labels={"profile": profile_label, "fit": model_options["label"]})
                    result = fit_and_plot((low, high, training.eval(axis).to_numpy(), training.eval(spec["y"]).to_numpy(),
                        axis, spec["y"], str(output / "fits" / axis / name), formula, model, spec["y_type"], bins,
                        axis_options["label"], spec["y_title"], spec["title"],
                        training.calibration_weight.to_numpy(), parameters[0]),
                        model_style, cfg["min_entries_per_bin"], fit_range=fit_range)
                    values = [result["parameters"][parameter] for parameter in parameters]
                    profile = measured["pileup_only_heldout"]
                    prediction = function(profile.x.to_numpy(), *values)
                    scores = {}
                    for domain, mask in (("fit_range", profile.within_fit_range),
                                         ("extrapolation", ~profile.within_fit_range)):
                        included = mask & np.isfinite(profile.value) & np.isfinite(prediction)
                        residual = profile.value[included].to_numpy() - prediction[included]
                        scores[f"heldout_bins_{domain}"] = int(included.sum())
                        scores[f"heldout_rmse_{domain}"] = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
                    results.append(dict(result, coefficient=name, axis=axis, model=model,
                                        function=formula, diagnostic_only=True, **scores))
                    grid = np.linspace(bins[0], bins[-1], 500)
                    prediction = function(grid, *values)
                    inside = (grid >= fit_range[0]) & (grid <= fit_range[1])
                    line, = axes[0].plot(grid, np.where(inside, prediction, np.nan), label=model_options["label"],
                                         color=model_options.get("color"))
                    axes[0].plot(grid, np.where(~inside, prediction, np.nan), linestyle="--", color=line.get_color())
                labels = cfg.get("comparison_labels", {})
                for stage, label, marker, color in (
                        ("current_corrected", labels.get("current", "Existing calibration"), "s", "#D55E00"),
                        ("candidate_corrected", labels.get("candidate", "Zprime refit"), "o", "#0072B2")):
                    profile = measured[stage]
                    axes[1].plot(profile.x, profile.value, marker=marker, color=color, label=label)
                axes[0].set_ylabel(options.get("label", spec["y_title"]))
                axes[1].set(ylabel=cfg["observables"][observable]["label"], xlabel=axis_options["label"],
                            xlim=(bins[0], bins[-1]))
                axes[1].axhline(0, color="0.6", linewidth=.7)
                add_legend(axes[0], style, title=(f"{cfg['dataset']} | {low:g} < |eta| < {high:g}\n"
                    f"Fit: {fit_range[0]:g}–{fit_range[1]:g} GeV; dashed: extrapolation"))
                add_legend(axes[1], style, title=f"Held-out {quantile}\nApplied calibration uses reco pT")
                save_plot(fig, directory / f"profile_eta_{low}_{high}.pdf", style)
    if profiles:
        pd.concat(profiles, ignore_index=True).to_csv(output / "diagnostic_profiles.csv", index=False)
        (output / "diagnostic_fits.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def run(config_path, inputdir, correction_lib, outputdir):
    config_path, correction_lib = Path(config_path).resolve(), Path(correction_lib).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    evaluation, base, base_path = load_id_config(config_path.parent / cfg["base_config"])
    dataset = evaluation["datasets"][cfg["dataset"]]
    definitions = base["workflow"]["InspectDistribution"]["define_variable"]
    candidate_definitions = dict(definitions, **cfg.get("candidate_definitions", {}))
    recipes = base["workflow"]["DeriveEffectiveArea"]["EffectiveArea"]
    payload = read_json(correction_lib)
    original = {item["name"]: item for item in payload["corrections"]}
    bins = edges(cfg["pt_binning"])
    fit_options = (dict.fromkeys(cfg["fits"], {}) if isinstance(cfg["fits"], list) else cfg["fits"])
    fits, targets = {}, set()
    for name, options in fit_options.items():
        if name not in original or name not in recipes:
            raise ValueError(f"Missing calibration coefficient or pipeline fit recipe: {name}")
        spec = dict(recipes[name], **options)
        if spec["x"] != "pt" or spec["y_type"].lower() == "none":
            raise ValueError(f"{name}: this comparison requires a binned pT fit")
        parameters = parse_fitted_function(spec["fitted_func"])[1]
        if len(parameters) > 2 and "coefficients" not in spec:
            raise ValueError(f"{name}: a higher-order fit needs an explicit coefficients mapping")
        coefficients = spec.get("coefficients", {name: spec.get("coefficient", "A1")})
        if not coefficients or set(coefficients.values()) - set(parameters):
            raise ValueError(f"{name}: coefficients must map correction names to fitted parameters")
        if targets.intersection(coefficients):
            raise ValueError("Multiple fits cannot write the same correction coefficient")
        targets.update(coefficients)
        fits[name] = (spec, coefficients)
    for name, options in cfg.get("profile_diagnostics", {}).items():
        if name not in fits or not options.get("axes") or not options.get("models"):
            raise ValueError(f"{name}: profile diagnostics require a configured fit, axes and models")
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
    for name, (spec, coefficients) in fits.items():
        values, eta_bins = {key: [] for key in coefficients}, spec["eta_bins"]
        for low, high in zip(eta_bins[:-1], eta_bins[1:]):
            mask = ((train.eta.abs() >= low) & (train.eta.abs() < high)
                    & (train.pt >= bins[0]) & (train.pt < bins[-1]))
            if spec.get("requirement"):
                mask &= train.eval(spec["requirement"])
            subset = train.loc[mask]
            y = subset.eval(spec["y"]).to_numpy()
            old_term = binned_profile(subset.pt, subset[name], subset.calibration_weight, bins, "mean",
                                      cfg["min_entries_per_bin"])
            style = dict(cfg.get("plot_style") or {})
            style["fit_labels"] = dict(style.get("fit_labels", {}), **spec.get("fit_labels", {}))
            reference_label = spec.get("reference_label", f"Existing correctionlib: {name} × pT")
            old = subset[name].to_numpy()
            if len(old):
                old_value = (f"{old[0]:.4g}" if np.allclose(old, old[0], rtol=1e-12, atol=0)
                             else f"[{old.min():.4g}, {old.max():.4g}] (varies within eta bin)")
                reference_label += f"\n{name} = {old_value} (coefficient of pT)"
            result = fit_and_plot((low, high, subset.pt.to_numpy(), y, "pt", spec["y"], str(output / "fits" / "pt"),
                spec["fitted_func"], name, spec["y_type"], bins, spec["x_title"], spec["y_title"], spec["title"],
                subset.calibration_weight.to_numpy(), next(iter(coefficients.values()))),
                style, cfg["min_entries_per_bin"],
                reference=dict(x=old_term.x, y=old_term.value * old_term.x,
                               label=reference_label), fit_range=spec.get("fit_range"))
            for target, parameter in coefficients.items():
                value = result["parameters"][parameter]
                values[target].append(value)
                old = subset[target].to_numpy() if target in subset else np.zeros(len(subset))
                new = float(applied_coefficient(target, value, candidate_definitions)[0])
                constant = np.allclose(old, old[0], rtol=1e-12, atol=0)
                results.append(dict(result, coefficient=target, parameter=parameter, value=value,
                                    current_min=float(old.min()), current_max=float(old.max()),
                                    new_applied=new, scale_factor=new / old[0] if constant and old[0] != 0 else np.nan,
                                    delta_if_constant=new - old[0] if constant else np.nan))
            profile = binned_profile(subset.pt, y, subset.calibration_weight, bins, spec["y_type"], cfg["min_entries_per_bin"])
            fit_range = result["fit_range"] or [bins[0], bins[-1]]
            profile["used_in_fit"] = np.isfinite(profile.value) & profile.x.between(*fit_range)
            fit_profiles.append(profile.assign(coefficient=name, eta_min=low, eta_max=high))
        for target, fitted_values in values.items():
            item = next((item for item in candidate_payload["corrections"] if item["name"] == target), None)
            if item is None:
                item = dict(name=target, version=1, inputs=deepcopy(original[name]["inputs"]),
                            output=dict(name=target, type="real"))
                candidate_payload["corrections"].append(item)
            item["data"] = dict(nodetype="binning", input="eta", edges=list(map(float, eta_bins)), content=fitted_values, flow="clamp")
    candidate_path = output / "corrections_zprime_pt.json"
    # Validate the complete candidate before writing; all non-refitted corrections stay unchanged.
    correctionlib.CorrectionSet.from_string(json.dumps(candidate_payload))
    candidate_path.write_text(json.dumps(candidate_payload, indent=2) + "\n")
    candidate_recipe = dict(shared={key: deepcopy(value) for key, value in base.items() if key != "workflow"},
                            workflow=deepcopy(base["workflow"]), evaluation=deepcopy(evaluation))
    candidate_recipe["workflow"]["InspectDistribution"]["define_variable"] = candidate_definitions
    candidate_recipe["workflow"]["InspectDistribution"]["correction_lib"] = str(candidate_path)
    candidate_config = output / "candidate_id.yaml"
    candidate_config.write_text(yaml.safe_dump(relocate_config_paths(candidate_recipe, base_path.parent), sort_keys=False))
    pd.DataFrame(results).to_csv(output / "coefficient_comparison.csv", index=False)
    pd.concat(fit_profiles, ignore_index=True).to_csv(output / "fit_profiles.csv", index=False)
    candidate = valid.copy()
    apply_corrections(candidate, str(candidate_path))
    define_variables(candidate, candidate_definitions)
    profiles = compare_profiles(valid, candidate, cfg, base, output)
    diagnostics = profile_diagnostics(train, valid, candidate, cfg, fits, output)
    fit_directories = {axis: str(output / "fits" / axis) for axis in dict.fromkeys(
        ["pt", *(item["axis"] for item in diagnostics)])}
    report = dict(loading=loading, train_rows=len(train), heldout_rows=len(valid),
                  fit_parameters="Unweighted fit to populated profile bins; errors are fit covariance errors",
                  below_minimum_profile_bins=int(profiles.value.isna().sum()),
                  candidate_calibration=str(candidate_path),
                  candidate_id_config=str(candidate_config),
                  fit_plot_directories=fit_directories,
                  full_id_updated=False)
    if diagnostics:
        report["profile_diagnostics"] = dict(
            fits=str(output / "diagnostic_fits.json"), profiles=str(output / "diagnostic_profiles.csv"),
            diagnostic_only=True, calibration_inputs_unchanged=True,
            metric="Unweighted RMSE [GeV] over populated held-out quantile bins, not a significance test",
            selection=dataset["requirement"],
            electron_response_max_relative_pt=preparation.get("electron_response_max_relative_pt"))
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(results)[["coefficient", "eta_min", "eta_max", "current_min", "current_max", "new_applied", "scale_factor"]].to_string(index=False))
    if diagnostics:
        print("Diagnostic fits only; held-out profile RMSE [GeV] (not a significance test):")
        print(pd.DataFrame(diagnostics)[["axis", "eta_min", "eta_max", "model",
              "heldout_rmse_fit_range", "heldout_rmse_extrapolation"]].to_string(index=False))
    for axis, directory in fit_directories.items():
        print(f"{axis} fit plots: {directory}")
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
