#!/usr/bin/env python3
"""Export full scouting IDs in CorrectionSet.compound and a standalone application script."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import correctionlib

from workflow.IDCorrection import build_correction_set
from workflow.working_points import configured_wp_sources, load_configured_wps, load_id_config, print_id_table


def export_id(config_path, outputdir, training_dir=None, correction_lib=None):
    config_path = Path(config_path).resolve()
    config, base, base_path = load_id_config(config_path)
    preparation = base["workflow"]["InspectDistribution"]
    training_dir = Path(training_dir or base["workflow"]["TrainCutBasedID_TMVA"]["outputdir"]).resolve()
    calibration_path = Path(correction_lib or preparation["correction_lib"]).resolve()
    opener = gzip.open if calibration_path.suffix == ".gz" else open
    with opener(calibration_path, "rt") as handle:
        calibration_text = handle.read()
    calibration = json.loads(calibration_text)
    correctionlib.CorrectionSet.from_string(calibration_text)
    artifacts = {"configuration": config_path, "base_configuration": base_path, "calibration": calibration_path}
    wps_by_region = {}
    for region in config.get("regions", preparation["regions"]):
        wps_by_region[region] = {}
        for source, options in configured_wp_sources(base, config).items():
            options = {"path": options} if isinstance(options, str) else options
            path = Path(options["path"].format(training_dir=training_dir, region=region,
                                              config_dir=config_path.parent)).resolve()
            artifacts[f"{source}/{region}"] = path
            wps_by_region[region][source] = load_configured_wps(path, options)
    exported, metadata = build_correction_set(base, config, calibration, wps_by_region)
    payload = json.dumps(exported, indent=2, allow_nan=False) + "\n"
    correctionlib.CorrectionSet.from_string(payload)
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / name for name in ("scouting_id.json.gz", "manifest.json", "apply_id.py", "cutbased_id.md")]
    if any(path.exists() for path in destinations):
        raise FileExistsError(f"ID export already exists in {output}; use a new output directory")
    packed = gzip.compress(payload.encode(), mtime=0)
    manifest = dict(correctionset=destinations[0].name, sha256=hashlib.sha256(packed).hexdigest(),
                    evaluator="compound", corrections=metadata, requirement=None,
                    regions=config["correctionlib"]["region_definition"],
                    artifacts={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                               for name, path in artifacts.items()})
    destinations[0].write_bytes(packed)
    destinations[1].write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("apply_id.py"), destinations[2])
    definitions = dict(preparation.get("define_variable", {}), **config.get("validation", {}).get("define_variable", {}))
    print_id_table(wps_by_region, destinations[3], None, config["correctionlib"]["region_definition"],
                   definitions=definitions, calibration=calibration, include_usage=True)
    return destinations[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Validation YAML or standalone pipeline with an evaluation section")
    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--training-dir", help="Completed regional WP directory")
    parser.add_argument("--correction-lib", help="Completed calibration JSON")
    args = parser.parse_args()
    print(export_id(args.config, args.outputdir, args.training_dir, args.correction_lib))


if __name__ == "__main__":
    main()
