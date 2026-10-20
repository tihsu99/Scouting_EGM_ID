#!/usr/bin/env python3
"""Run the configured ID studies using the existing analysis stages."""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path

import yaml


STEPS = ("calibrate", "prepare", "train", "validate", "highpt")


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


def run(config_path, inputdir, outputdir, steps=None, studies=None, manifest=None, check_only=False):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    steps = cfg.get("steps", STEPS) if steps is None else steps
    if not isinstance(steps, (list, tuple)) or not steps or any(step not in STEPS for step in steps):
        raise ValueError(f"steps must be a nonempty list drawn from {STEPS}")
    base_path = (config_path.parent / cfg["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text())
    workflow = deepcopy(base["workflow"])
    for module, overrides in cfg.get("workflow_overrides", {}).items():
        workflow[module].update(overrides)
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
    outputdir.mkdir(parents=True, exist_ok=True)
    resolved = dict(configuration=cfg, base=base, inputdir=shared["inputdir"])
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
        data_dir = study_dir / "data"
        train_dir = study_dir / "train_results"
        process = {category: {"samples": dict.fromkeys(study[category], 1.0),
                              "requirement": cfg[f"{category}_requirement"]}
                   for category in ("signal", "background")}
        if "prepare" in steps:
            stage("InspectDistribution", process=process, storedir=str(data_dir),
                  plotdir=str(study_dir / "distributions"), correction_lib=str(corrections),
                  training_weight_mode=cfg["training_weight_mode"])
        if "train" in steps:
            stage("TrainCutBasedID_TMVA", inputdir=str(data_dir), outputdir=str(train_dir))
        wp_sources = {f"Optimized({tag})": str(train_dir / "{region}" / f"CutBased_WPs_{tag}_TMVA_seq.json")
                      for tag in workflow["TrainCutBasedID_TMVA"]["variables"]}
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
    (outputdir / "last_run.json").write_text(json.dumps({"steps": list(steps), "studies": list(selected)}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--inputdir", required=True, help="Mounted campaign directory containing sample folders")
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--manifest", help="Production farm manifest; verifies completeness and receipt provenance")
    parser.add_argument("--studies", nargs="+")
    parser.add_argument("--steps", nargs="+", choices=STEPS,
                        help="Override YAML steps; defaults to all stages when YAML omits steps")
    parser.add_argument("--check-only", action="store_true", help="Verify normalization and inputs without fitting or training")
    args = parser.parse_args()
    run(args.config, args.inputdir, args.outputdir, args.steps, args.studies, args.manifest, args.check_only)


if __name__ == "__main__":
    main()
