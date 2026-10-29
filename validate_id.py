#!/usr/bin/env python3
"""Evaluate existing cut-based WPs with the preparation and validation stages."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import multiprocessing
from pathlib import Path
import time

import yaml

from apply_id import id_evaluators, id_inputs
from workflow.InspectDistribution import InspectDistributionAnalysis
from workflow.ValidateID import ValidateIDAnalysis
from workflow.working_points import configured_wp_sources, load_id_config
from workflow.reference_ids import load_reference_recipes, load_correctionlib_references
from workflow.exported_ids import load_exported_wps


def _plot_job(name, arguments, exported_spec=None):
    """Run one comparison/region without sharing Matplotlib or native ID objects."""
    import matplotlib
    matplotlib.use("Agg")
    started = time.perf_counter()
    print(f"[plot] Starting {name}", flush=True)
    if exported_spec is not None:
        arguments["exported_id"] = load_exported_wps(*exported_spec)
    ValidateIDAnalysis(**arguments)
    print(f"[plot] Finished {name} in {time.perf_counter() - started:.1f} s", flush=True)


def run(config_path, outputdir, inputdir=None, prepared_dir=None, training_dir=None,
        correction_lib=None, steps=("prepare", "plot"), id_correctionlib=None, workers=None):
    config_path = Path(config_path).resolve()
    cfg, base, _ = load_id_config(config_path)
    workers = cfg.get("workers", 1) if workers is None else workers
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    cfg["workers"] = workers
    for name, overrides in cfg.pop("dataset_overrides", {}).items():
        if name not in cfg["datasets"]:
            raise ValueError(f"Unknown dataset override: {name}")
        cfg["datasets"][name].update(overrides)
    preparation = deepcopy(base["workflow"]["InspectDistribution"])
    correctionlib_only = bool(id_correctionlib and training_dir is None)
    raw_id_only = correctionlib_only and correction_lib is None
    prepared_root = prepared_dir or preparation.get("storedir")
    paths = dict(inputdir=str(Path(inputdir or base["inputdir"]).resolve()),
                 prepared_dir=str(Path(prepared_root or Path(outputdir) / "evaluation_data").resolve()),
                 training_dir=None if correctionlib_only else str(Path(
                     training_dir or base["workflow"]["TrainCutBasedID_TMVA"]["outputdir"]).resolve()),
                 outputdir=str(Path(outputdir).resolve()), config_dir=str(config_path.parent), region="{region}")
    correction_lib = None if raw_id_only else str(Path(correction_lib or preparation["correction_lib"]).resolve())
    regions = cfg.get("regions", preparation["regions"])
    sources = {} if correctionlib_only else configured_wp_sources(base, cfg)
    exported_id = None
    exported_spec = None
    if correctionlib_only:
        needs_components = any(dict(cfg["validation"], **comparison.get("validation", {})).get("nminus1_efficiencies", False)
                               for comparison in cfg["comparisons"].values())
        if raw_id_only and needs_components:
            raise ValueError("N-minus-one feature labels require --correction-lib and the matching feature YAML")
        exported_spec = (id_correctionlib, base, cfg, correction_lib, needs_components)
        exported_id = load_exported_wps(*exported_spec)
    if raw_id_only:
        raw_columns = list(dict.fromkeys([*preparation["input_columns"],
            *(item.name for wp in id_evaluators(exported_id[0]).values() for item in id_inputs(wp))]))
        for comparison in cfg["comparisons"].values():
            validation = dict(cfg["validation"], **comparison.get("validation", {}))
            missing = set(validation["monitor_eff"]) - set(raw_columns)
            if missing or validation.get("variables"):
                raise ValueError("Derived-variable plots require --correction-lib and the matching feature YAML; "
                                 f"unavailable raw monitors: {sorted(missing)}")
            comparison.setdefault("validation", {}).update(input_columns=["sample", "weight", *raw_columns],
                                                           define_variable={}, cut_distributions={})
    artifacts = {"correction_lib": correction_lib} if correction_lib else {}
    for source, item in load_reference_recipes(cfg.get("validation", {}).get("reference_sources"), config_path.parent).items():
        artifacts[f"reference/{source}"] = item["path"]
    for source, item in load_correctionlib_references(
            cfg.get("validation", {}).get("correctionlib_references"), config_path.parent).items():
        artifacts[f"correctionlib_reference/{source}"] = item["metadata"]["path"]
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
        same_export = (correctionlib_only and previous.get("artifacts", {}).get("id_correctionlib")
                       == current["artifacts"].get("id_correctionlib"))
        # Presentation changes can redraw cached evaluation data in the same directory.
        for settings in (previous, current):
            # Process count does not change candidates, weights or ID decisions.
            settings["configuration"].pop("workers", None)
            if same_export:
                # The same exported ID remains authoritative when dropping the training cross-check.
                settings["paths"].pop("training_dir", None)
                for source in cfg["wp_sources"]:
                    for region in regions:
                        settings["artifacts"].pop(f"{source}/{region}", None)
            # Export recipes do not change a plot; an applied export is fingerprinted above.
            settings["configuration"].pop("correctionlib", None)
            for field in ("cut_distributions", "nminus1_efficiencies", "monitor_eff"):
                settings["configuration"].get("validation", {}).pop(field, None)
                for comparison in settings["configuration"].get("comparisons", {}).values():
                    comparison.get("validation", {}).pop(field, None)
                settings["training_configuration"].get("workflow", {}).get("ValidateID", {}).pop(field, None)
            trainer = settings["training_configuration"].get("workflow", {}).get("TrainCutBasedID_TMVA", {})
            for field in ("plot_samples", "cut_plot_variables"):
                trainer.pop(field, None)
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
    if raw_id_only:
        shared.update(input_columns=raw_columns, define_variable={})
    datasets = deepcopy(cfg["datasets"])
    used = set()
    for comparison in cfg["comparisons"].values():
        used.update([comparison["signal"], *comparison["backgrounds"]])
    for name, dataset in datasets.items():
        if name not in used:
            continue
        inputs = dataset.get("path", [])
        inputs = inputs if isinstance(inputs, list) else [inputs]
        from_raw = not inputs or (not prepared_root and any("{prepared_dir}" in path for path in inputs))
        if not from_raw:
            dataset["path"] = [path.format(**paths) for path in inputs]
        else:
            if not dataset.get("samples"):
                raise ValueError(f"Dataset {name} needs --prepared-dir or YAML samples for preparation from --inputdir")
            dataset["path"] = []
            # Prepare one sample at a time; QCD has many independent generation bins.
            for sample in dataset["samples"]:
                data_dir = output / "evaluation_data" / name / sample
                dataset["path"].append(str(data_dir / "{region}" / "valid" / f"{name}.parquet"))
                if "prepare" in steps:
                    arguments = dict(shared, process={name: {
                        "samples": {sample: 1.0}, "requirement": dataset["requirement"]}},
                        validation_split=dataset.get("validation_split", "all"),
                        storedir=str(data_dir), plotdir=str(output / "preparation" / name / sample))
                    InspectDistributionAnalysis(**arguments)
        for region in regions:
            for template in dataset["path"]:
                path = Path(template.format(region=region))
                if not path.is_file():
                    if from_raw:
                        recovery = "Run with --steps prepare plot to build the evaluation cache from --inputdir."
                    elif dataset.get("samples") and any("{prepared_dir}" in item for item in inputs):
                        recovery = ("Point --prepared-dir at existing regional validation tables, or omit it "
                                    "and use --steps prepare plot with a new --outputdir to prepare from --inputdir.")
                    else:
                        recovery = "Check the configured dataset path and --prepared-dir."
                    raise FileNotFoundError(f"Missing validation table for {name}: {path}. {recovery}")

    if "plot" in steps:
        jobs = []
        for name, comparison in cfg["comparisons"].items():
            selected = {key: dict(datasets[key], is_signal=key == comparison["signal"])
                        for key in [comparison["signal"], *comparison["backgrounds"]]}
            plot_style = deepcopy(cfg.get("plot_style", base.get("plot_style")) or {})
            for key in ("roc_outputdir", "roc_filename_prefix"):
                if key in plot_style:
                    plot_style[key] = plot_style[key].format(comparison=name, **paths)
            validation = dict(cfg["validation"], **comparison.get("validation", {}))
            arguments = dict(inputdir=paths["prepared_dir"], outputdir=str(output / name),
                               config_dir=str(config_path.parent),
                               datasets=selected, wp_sources=sources,
                               region_definition=base["region_definition"],
                               plot_style=plot_style,
                               id_correctionlib=artifacts.get("id_correctionlib"),
                               label=comparison.get("label", name), **validation)
            jobs.extend((f"{name}/{region}", dict(arguments, regions=[region])) for region in regions)
        active_workers = min(workers, len(jobs))
        print(f"[plot] {len(jobs)} comparison/region jobs; {active_workers} worker(s)", flush=True)
        if active_workers > 1:
            roc_targets = set()
            for name, arguments in jobs:
                style = arguments["plot_style"]
                target = (str(Path(style.get("roc_outputdir", Path(arguments["outputdir"]) / arguments["regions"][0])).resolve()),
                          style.get("roc_filename_prefix", ""), arguments["regions"][0])
                if target in roc_targets:
                    raise ValueError(f"Parallel ROC outputs overlap for {name}; use distinct roc_outputdir "
                                     "or roc_filename_prefix values, e.g. '{comparison}_'")
                roc_targets.add(target)
            # Spawn avoids inheriting PyArrow/Matplotlib thread state from preparation.
            with ProcessPoolExecutor(max_workers=active_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                futures = {pool.submit(_plot_job, name, arguments, exported_spec): name for name, arguments in jobs}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        for pending in futures:
                            pending.cancel()
                        raise RuntimeError(f"Validation plotting failed for {futures[future]}: {error}") from error
        else:
            for name, arguments in jobs:
                _plot_job(name, dict(arguments, exported_id=exported_id))
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--inputdir", help="Raw sample folders; defaults to base_config inputdir")
    parser.add_argument("--prepared-dir", help="Existing held-out data; uses base_config storedir when set, "
                        "otherwise prepares YAML samples under outputdir/evaluation_data")
    parser.add_argument("--training-dir", help="WP directory; omit with --id-correctionlib to use only the exported ID")
    parser.add_argument("--correction-lib", help="Calibration for derived features; omit with --id-correctionlib for raw-input full-ID plots")
    parser.add_argument("--id-correctionlib", help="Full ID JSON; --training-dir optionally enables the training-WP cross-check")
    parser.add_argument("--steps", nargs="+", choices=("prepare", "plot"), default=["prepare", "plot"])
    parser.add_argument("--workers", type=int, help="Plot processes, one per comparison/region; defaults to YAML workers or 1")
    args = parser.parse_args()
    run(args.config, args.outputdir, args.inputdir, args.prepared_dir, args.training_dir,
        args.correction_lib, args.steps, args.id_correctionlib, args.workers)


if __name__ == "__main__":
    main()
