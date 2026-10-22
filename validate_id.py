#!/usr/bin/env python3
"""Evaluate existing cut-based WPs with the preparation and validation stages."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import yaml

from workflow.InspectDistribution import InspectDistributionAnalysis
from workflow.ValidateID import ValidateIDAnalysis


def run(config_path, outputdir, inputdir=None, prepared_dir=None, training_dir=None,
        correction_lib=None, steps=("prepare", "plot"), id_correctionlib=None):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    base_path = (config_path.parent / cfg["base_config"]).resolve()
    base = yaml.safe_load(base_path.read_text())
    preparation = deepcopy(base["workflow"]["InspectDistribution"])
    paths = dict(inputdir=str(Path(inputdir or base["inputdir"]).resolve()),
                 prepared_dir=str(Path(prepared_dir or preparation["storedir"]).resolve()),
                 training_dir=str(Path(training_dir or base["workflow"]["TrainCutBasedID_TMVA"]["outputdir"]).resolve()),
                 outputdir=str(Path(outputdir).resolve()), config_dir=str(config_path.parent), region="{region}")
    correction_lib = str(Path(correction_lib or preparation["correction_lib"]).resolve())
    regions = cfg.get("regions", preparation["regions"])
    sources = deepcopy(cfg["wp_sources"])
    artifacts = {"correction_lib": correction_lib}
    if id_correctionlib:
        artifacts["id_correctionlib"] = str(Path(id_correctionlib).resolve())
    for source, options in sources.items():
        options["path"] = options["path"].format(**paths)
        for region in regions:
            artifacts[f"{source}/{region}"] = options["path"].format(region=region)
    # Check the completed artifacts before preparing independent samples.
    fingerprints = {name: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                    for name, path in artifacts.items()}
    output = Path(paths["outputdir"])
    resolved = dict(configuration=cfg, training_configuration=base, paths=paths, artifacts=fingerprints)
    saved = output / "resolved_config.yaml"
    if saved.exists():
        previous, current = yaml.safe_load(saved.read_text()), deepcopy(resolved)
        # Presentation changes can redraw cached evaluation data in the same directory.
        for settings in (previous, current):
            # Export recipes do not change a plot; an applied export is fingerprinted above.
            settings["configuration"].pop("correctionlib", None)
            for key in ("configuration", "training_configuration"):
                settings[key].pop("plot_style", None)
            for group, field in (("wp_sources", "labels"), ("datasets", "label"), ("comparisons", "label")):
                for options in settings["configuration"].get(group, {}).values():
                    if isinstance(options, dict):
                        options.pop(field, None)
        if previous != current:
            raise ValueError("Configuration or WP/correction artifacts changed: use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    saved.write_text(yaml.safe_dump(resolved, sort_keys=False))

    shared = {key: deepcopy(value) for key, value in base.items() if key not in ("workflow", "process")}
    shared.update(preparation)
    shared.update(cfg.get("preparation", {}))
    shared.update(inputdir=paths["inputdir"], correction_lib=correction_lib, regions=regions,
                  normalization={}, validation_only=True, kinematic_reweighting=None,
                  stitch_binning=None, inspect_variable={})
    datasets = deepcopy(cfg["datasets"])
    used = set()
    for comparison in cfg["comparisons"].values():
        used.update([comparison["signal"], *comparison["backgrounds"]])
    for name, dataset in datasets.items():
        if name not in used:
            continue
        if "path" in dataset:
            inputs = dataset["path"] if isinstance(dataset["path"], list) else [dataset["path"]]
            dataset["path"] = [path.format(**paths) for path in inputs]
        else:
            dataset["path"] = []
            # Prepare one sample at a time; QCD has many independent generation bins.
            for sample in dataset["samples"]:
                data_dir = output / "evaluation_data" / name / sample
                dataset["path"].append(str(data_dir / "{region}" / "valid" / f"{name}.parquet"))
                if "prepare" in steps:
                    arguments = dict(shared, process={name: {
                        "samples": {sample: 1.0}, "requirement": dataset["requirement"]}},
                        storedir=str(data_dir), plotdir=str(output / "preparation" / name / sample))
                    InspectDistributionAnalysis(**arguments)
        for region in regions:
            for template in dataset["path"]:
                path = Path(template.format(region=region))
                if not path.is_file():
                    raise FileNotFoundError(f"Missing validation table for {name}: {path}")

    if "plot" in steps:
        for name, comparison in cfg["comparisons"].items():
            selected = {key: dict(datasets[key], is_signal=key == comparison["signal"])
                        for key in [comparison["signal"], *comparison["backgrounds"]]}
            plot_style = deepcopy(cfg.get("plot_style", base.get("plot_style")) or {})
            for key in ("roc_outputdir", "roc_filename_prefix"):
                if key in plot_style:
                    plot_style[key] = plot_style[key].format(comparison=name, **paths)
            ValidateIDAnalysis(inputdir=paths["prepared_dir"], outputdir=str(output / name),
                               datasets=selected, regions=regions, wp_sources=sources,
                               region_definition=base["region_definition"],
                               plot_style=plot_style,
                               id_correctionlib=artifacts.get("id_correctionlib"),
                               label=comparison.get("label", name), **cfg["validation"])
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--inputdir", help="Independent sample folders; defaults to base_config inputdir")
    parser.add_argument("--prepared-dir", help="Existing held-out data; defaults to base_config storedir")
    parser.add_argument("--training-dir", help="Completed WP directory; defaults to base_config training outputdir")
    parser.add_argument("--correction-lib", help="Completed correction JSON; defaults to base_config correction_lib")
    parser.add_argument("--id-correctionlib", help="Full ID JSON: check equivalence and plot its raw-input decisions")
    parser.add_argument("--steps", nargs="+", choices=("prepare", "plot"), default=["prepare", "plot"])
    args = parser.parse_args()
    run(args.config, args.outputdir, args.inputdir, args.prepared_dir, args.training_dir,
        args.correction_lib, args.steps, args.id_correctionlib)


if __name__ == "__main__":
    main()
