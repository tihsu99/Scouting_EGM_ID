#!/usr/bin/env python3
"""Run the configured ID studies using the existing analysis stages."""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path

import yaml


STEPS = ("calibrate", "prepare", "train", "validate", "highpt")


def preparation_recipe(resolved, study):
    """Compare the settings that produced prepared tables, excluding later stages."""
    cfg, base = resolved["configuration"], resolved["base"]
    workflow = deepcopy(cfg["workflow"] if "workflow" in cfg else base["workflow"])
    for module, overrides in cfg.get("workflow_overrides", {}).items():
        workflow[module].update(overrides)
    arguments = {key: value for key, value in base.items() if key not in ("workflow", "process")}
    arguments.update(inputdir=resolved["inputdir"], normalization=cfg["normalization"])
    arguments.update(cfg.get("shared", {}))
    arguments.update(workflow["InspectDistribution"])
    arguments.update(training_weight_mode=cfg["training_weight_mode"],
                     correction_lib=cfg.get("correction_lib"),
                     process={category: {"samples": dict.fromkeys(cfg["studies"][study][category], 1.0),
                                         "requirement": cfg[f"{category}_requirement"]}
                              for category in ("signal", "background")})
    return arguments


def check_manifest(path, inputdir, samples):
    jobs = json.loads(Path(path).read_text())
    found = set()
    expected = set()
    for job in jobs:
        if job["sample"] not in samples:
            continue
        found.add(job["sample"])
        output = Path(inputdir) / job["sample"] / Path(job["output_url"]).name
        receipt_path = Path(str(output) + ".json")
        if not output.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Incomplete production: {output}")
        receipt = json.loads(receipt_path.read_text())
        if receipt["input_files"] != job["input_files"] or receipt["dataset"] != job["dataset"]:
            raise ValueError(f"Receipt does not match production manifest: {output}")
        expected.add(output.resolve())
    if found != samples:
        raise ValueError(f"Samples missing from production manifest: {samples - found}")
    actual = {p.resolve() for sample in samples for p in (Path(inputdir) / sample).glob("*.parquet")}
    if actual != expected:
        raise ValueError("Input directory contains files outside the selected production manifest")


def run(config_path, inputdir, outputdir, steps=None, studies=None, manifest=None, check_only=False,
        correction_lib=None, prepared_dir=None, preview_training=False, regions=None):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    if correction_lib is not None:
        cfg["correction_lib"] = str(Path(correction_lib).expanduser().resolve())
    steps = cfg.get("steps", STEPS) if steps is None else steps
    if not isinstance(steps, (list, tuple)) or not steps or any(step not in STEPS for step in steps):
        raise ValueError(f"steps must be a nonempty list drawn from {STEPS}")
    if prepared_dir is not None and (check_only or any(step not in ("train", "validate") for step in steps)):
        raise ValueError("--prepared-dir supports only train/validate, without --check-only")
    if preview_training and (check_only or list(steps) != ["train"]):
        raise ValueError("--preview-training requires --steps train, without --check-only")
    if "workflow" in cfg and "base_config" in cfg:
        raise ValueError("Use an inline workflow or base_config, not both")
    base = {}
    if "base_config" in cfg:
        base_path = (config_path.parent / cfg["base_config"]).resolve()
        base = yaml.safe_load(base_path.read_text())
    workflow = deepcopy(cfg["workflow"] if "workflow" in cfg else base["workflow"])
    for module, overrides in cfg.get("workflow_overrides", {}).items():
        workflow[module].update(overrides)
    if regions is not None:
        if check_only or any(step not in ("train", "validate") for step in steps):
            raise ValueError("--regions supports only train/validate, without --check-only")
        if (not isinstance(regions, (list, tuple)) or not regions
                or any(not isinstance(region, str) for region in regions)
                or len(set(regions)) != len(regions)):
            raise ValueError("regions must be a nonempty list of unique configured region names")
        regions = list(regions)
        for step in steps:
            module = "TrainCutBasedID_TMVA" if step == "train" else "ValidateID"
            unknown = set(regions) - set(workflow[module]["regions"])
            if unknown:
                raise ValueError(f"Unknown regions for {step}: {sorted(unknown)}")
            workflow[module]["regions"] = regions
    selected = cfg["studies"] if studies is None else {name: cfg["studies"][name] for name in studies}
    outputdir = Path(outputdir).resolve()
    correction_dir = outputdir / "EffectiveArea"
    corrections = correction_dir / "corrections_all.json"
    if cfg.get("correction_lib"):
        if "calibrate" in steps:
            raise ValueError("Remove calibrate from steps to reuse correction_lib, or remove correction_lib to derive new corrections")
        corrections = (config_path.parent / cfg["correction_lib"]).resolve()
    if "calibrate" not in steps and any(step in steps for step in ("prepare", "highpt")):
        if not corrections.is_file():
            raise FileNotFoundError(f"Missing completed calibration: {corrections}")
    shared = {key: value for key, value in base.items() if key not in ("workflow", "process")}
    shared.update(inputdir=str(Path(inputdir).resolve()), normalization=cfg["normalization"])
    shared.update(cfg.get("shared", {}))
    samples = set(cfg["calibration_samples"])
    for study in selected.values():
        samples.update(study["signal"])
        samples.update(study["background"])
    if "highpt" in steps:
        samples.update(cfg["high_pt"]["signal"])
    if manifest:
        check_manifest(manifest, shared["inputdir"], samples)
    if check_only:
        from workflow.data import load_samples
        _, report = load_samples(shared["inputdir"], {"all": {"samples": dict.fromkeys(sorted(samples), 1.0)}},
                                 shared.get("n_files", -1), shared["normalization"])
        print(json.dumps(report, indent=2))
        return
    resolved = dict(configuration=cfg, base=base, inputdir=shared["inputdir"])
    if regions is not None:
        resolved["regions"] = regions
    if prepared_dir is not None:
        prepared_dir = Path(prepared_dir).expanduser().resolve()
        previous = yaml.safe_load((prepared_dir / "resolved_config.yaml").read_text())
        if previous.get("prepared_dir"):
            raise ValueError("--prepared-dir must point to the original prepare output directory")
        for name in selected:
            if name not in previous["configuration"]["studies"] or preparation_recipe(previous, name) != preparation_recipe(resolved, name):
                raise ValueError(f"Prepared data settings differ for {name}; restore the original preparation settings "
                                 "or rerun prepare before reusing these tables")
            required_regions = {region for step in steps for region in workflow[
                "TrainCutBasedID_TMVA" if step == "train" else "ValidateID"]["regions"]}
            for region in required_regions:
                for split in ("train", "valid"):
                    for role in ("signal", "background"):
                        path = prepared_dir / name / "data" / region / split / f"{role}.parquet"
                        if not path.is_file():
                            raise FileNotFoundError(f"Missing prepared table: {path}")
        resolved["prepared_dir"] = str(prepared_dir)
    outputdir.mkdir(parents=True, exist_ok=True)
    saved = outputdir / "resolved_config.yaml"
    if saved.exists() and yaml.safe_load(saved.read_text()) != resolved:
        raise ValueError("Configuration changed: use a new output directory to avoid mixing artifacts")
    saved.write_text(yaml.safe_dump(resolved, sort_keys=False))

    def stage(name, **overrides):
        arguments = dict(shared)
        arguments.update(deepcopy(workflow[name]))
        arguments.update(overrides)
        module = importlib.import_module(f"workflow.{name}")
        return getattr(module, f"{name}Analysis")(**arguments)

    if "calibrate" in steps:
        calibration = {"signal": {"samples": dict.fromkeys(cfg["calibration_samples"], 1.0),
                                   "requirement": cfg["signal_requirement"]}}
        stage("DeriveEffectiveArea", process=calibration, plotdir=str(correction_dir),
              requirement=cfg["signal_requirement"], calibration_train_only=True,
              calibration_weight_mode=cfg["training_weight_mode"])
    for name, study in selected.items():
        study_dir = outputdir / name
        data_dir = (prepared_dir / name if prepared_dir is not None else study_dir) / "data"
        train_dir = study_dir / "train_results"
        process = {category: {"samples": dict.fromkeys(study[category], 1.0),
                              "requirement": cfg[f"{category}_requirement"]}
                   for category in ("signal", "background")}
        if "prepare" in steps:
            stage("InspectDistribution", process=process, storedir=str(data_dir),
                  plotdir=str(study_dir / "distributions"), correction_lib=str(corrections),
                  training_weight_mode=cfg["training_weight_mode"])
        if "train" in steps:
            stage("TrainCutBasedID_TMVA", inputdir=str(data_dir), outputdir=str(train_dir),
                  preview_training=preview_training)
        wp_sources = {f"Optimized({tag})": str(train_dir / "{region}" / f"CutBased_WPs_{tag}_TMVA_seq.json")
                      for tag in workflow["TrainCutBasedID_TMVA"]["variables"]}
        if cfg.get("default_wp"):
            reference = (config_path.parent / cfg["default_wp"]).resolve()
            wp_sources["Default"] = str(reference)
        if "validate" in steps:
            stage("ValidateID", inputdir=str(data_dir), outputdir=str(study_dir / "performance"), wp_sources=wp_sources)
        if "highpt" in steps:
            high = cfg["high_pt"]
            low, upper = high["range"]
            process["signal"]["samples"] = dict.fromkeys(high["signal"], 1.0)
            process["signal"]["requirement"] += f" & (pt >= {low}) & (pt < {upper})"
            high_dir = study_dir / "highpt"
            # Only independent Z' is prepared here; reuse the held-out background.
            stage("InspectDistribution", process={"signal": process["signal"]},
                  storedir=str(high_dir / "data"), plotdir=str(high_dir / "distributions"),
                  correction_lib=str(corrections), validation_only=True, stitch_binning=None,
                  inspect_variable={}, kinematic_reweighting=None)
            import pandas as pd
            for region in workflow["ValidateID"]["regions"]:
                frame = pd.read_parquet(data_dir / region / "valid" / "background.parquet")
                frame = frame.loc[(frame.pt >= low) & (frame.pt < upper)]
                frame.to_parquet(high_dir / "data" / region / "valid" / "background.parquet", index=False)
            stage("ValidateID", inputdir=str(high_dir / "data"), outputdir=str(high_dir / "performance"),
                  wp_sources=wp_sources, pt_regions={f"pt{low}to{upper}": [low, upper]},
                  monitor_eff={"pt": {"binning": high["binning"], "label": high.get("label", "pt")}}, variables={})
    (outputdir / "last_run.json").write_text(json.dumps({"steps": list(steps), "studies": list(selected),
                                                       "preview_training": preview_training}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--inputdir", required=True, help="Mounted campaign directory containing sample folders")
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--manifest", help="Production farm manifest; verifies completeness and receipt provenance")
    parser.add_argument("--correction-lib", help="Reuse a completed corrections_all.json; keep this override for subsequent stages in the same output directory")
    parser.add_argument("--prepared-dir", help="Reuse an existing postprocess output root containing <study>/data; "
                        "train/validate only, with a new --outputdir for changed training settings")
    parser.add_argument("--preview-training", action="store_true", help="Sample prepared training candidates and "
                        "report quantile bounds without running TMVA; requires --steps train")
    parser.add_argument("--studies", nargs="+")
    parser.add_argument("--regions", nargs="+", help="Select configured regions for train/validate, e.g. Barrel or Endcap; "
                        "use separate output directories for parallel region jobs")
    parser.add_argument("--steps", nargs="+", choices=STEPS,
                        help="Override YAML steps; defaults to all stages when YAML omits steps")
    parser.add_argument("--check-only", action="store_true", help="Verify normalization and inputs without fitting or training")
    args = parser.parse_args()
    run(args.config, args.inputdir, args.outputdir, args.steps, args.studies, args.manifest, args.check_only,
        correction_lib=args.correction_lib, prepared_dir=args.prepared_dir, preview_training=args.preview_training,
        regions=args.regions)


if __name__ == "__main__":
    main()
