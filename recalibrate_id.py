#!/usr/bin/env python3
"""Update an exported ID's calibrated features while preserving its original cuts."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

import correctionlib
import yaml

from compare_isolation_pt import relocate_config_paths
from workflow.exported_ids import _expanded_formula, load_exported_wps, read_json
from workflow.IDCorrection import id_expression_compiler
from workflow.working_points import load_id_config


def recalibrate_id(config_path, id_correctionlib, correction_lib, candidate_config, outputdir):
    config_path, candidate_config = Path(config_path).resolve(), Path(candidate_config).resolve()
    config, base, base_path = load_id_config(config_path)
    candidate, new_base, new_base_path = load_id_config(candidate_config)
    new_calibration = Path(new_base["workflow"]["InspectDistribution"]["correction_lib"]).resolve()
    output = Path(outputdir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")

    # Match the original feature expressions, including calibration, before changing anything.
    _, templates = load_exported_wps(id_correctionlib, base, config, correction_lib, nminus1=True)
    old_compiler = id_expression_compiler(base, config, read_json(correction_lib))
    new_compiler = id_expression_compiler(new_base, candidate, read_json(new_calibration))
    if old_compiler.inputs != new_compiler.inputs:
        raise ValueError("Recalibration must preserve the original raw input definitions")
    if (base["region_definition"] != new_base["region_definition"] or
            old_compiler.compile(config["correctionlib"]["requirement"]) !=
            new_compiler.compile(candidate["correctionlib"]["requirement"])):
        raise ValueError("Recalibration must preserve the original regions and fiducial selection")

    replacements = {}
    definitions = dict(base["workflow"]["InspectDistribution"].get("define_variable", {}),
                       **config.get("validation", {}).get("define_variable", {}))
    for variable in definitions.keys() & new_compiler.symbols.keys():
        old_feature = old_compiler.symbols[variable].replace("e+", "e")
        new_feature = new_compiler.symbols[variable].replace("e+", "e")
        # Never replace bare calibration constants: they can also be cut thresholds.
        if old_feature != new_feature and old_compiler.used_inputs(old_feature):
            if old_feature in replacements and replacements[old_feature] != new_feature:
                raise ValueError(f"Ambiguous recalibration of aliased feature: {variable}")
            replacements[old_feature] = new_feature
    if not replacements:
        raise ValueError("No calibrated input-dependent features changed")
    replacement_pattern = re.compile("|".join(re.escape(item) for item in sorted(replacements, key=len, reverse=True)))

    payload = read_json(id_correctionlib)
    updated = deepcopy(payload)
    components = {item["name"]: item for item in updated["corrections"]}
    covered = {wp.meta["reference_evaluator"] for wps in templates.values() for wp in wps}
    if covered != {item["name"] for item in payload.get("compound_corrections", [])}:
        raise ValueError("The original YAML must identify every exported compound WP")
    symbols = {name: f"__raw_{index}__" for index, name in enumerate(old_compiler.inputs)}
    changes = []
    for wps in templates.values():
        for wp in wps:
            for variable, group in wp.meta["id_components"].items():
                old_feature = old_compiler.compile(f"`{variable}`").replace("e+", "e")
                new_feature = new_compiler.compile(f"`{variable}`").replace("e+", "e")
                if old_feature == new_feature:
                    continue
                for name in group:
                    if not re.fullmatch(r"__id__/.+/cut_\d+", name):
                        continue
                    component = components[name]
                    expression = _expanded_formula(component["data"], symbols)
                    if old_feature not in expression:
                        raise ValueError(f"{name}: original feature not found: {variable}")
                    # Update dependent intermediate finite checks too, without rewriting inserted expressions.
                    expression = replacement_pattern.sub(lambda match: replacements[match.group()], expression)
                    used = new_compiler.used_inputs(expression)
                    if set(used) != {item["name"] for item in component["inputs"]}:
                        raise ValueError(f"{name}: recalibration would change the raw input contract")
                    component["data"] = (new_compiler.factored_formula(expression) if len(used) > 4 else
                                         new_compiler.formula(expression)[1])
                    changes.append(dict(wp=wp.meta["reference_evaluator"], variable=variable, component=name))
    if not changes:
        raise ValueError("No exported ID features changed; check the candidate recipe and calibration")

    packed = gzip.compress((json.dumps(updated, indent=2, allow_nan=False) + "\n").encode(), mtime=0)
    correctionlib.CorrectionSet.from_string(gzip.decompress(packed).decode())
    artifacts = dict(original_id=Path(id_correctionlib).resolve(), original_config=config_path,
                     original_base_config=base_path, original_calibration=Path(correction_lib).resolve(),
                     candidate_config=candidate_config, candidate_base_config=new_base_path,
                     candidate_calibration=new_calibration)
    manifest = dict(correctionset="scouting_id.json.gz", sha256=hashlib.sha256(packed).hexdigest(),
                    evaluator="compound", changes=changes, thresholds_preserved=True,
                    artifacts={key: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                               for key, path in artifacts.items()})
    output.mkdir(parents=True, exist_ok=True)
    destination = output / manifest["correctionset"]
    destination.write_bytes(packed)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Keep the original evaluation sample/plot choices with the candidate's feature definitions.
    validation_config = deepcopy(config)
    validation_config["base_config"] = str(candidate_config)
    validation_config["validation"]["define_variable"] = candidate.get("validation", {}).get("define_variable", {})
    (output / "validation.yaml").write_text(yaml.safe_dump(
        relocate_config_paths(validation_config, config_path.parent), sort_keys=False))
    shutil.copyfile(Path(__file__).with_name("apply_id.py"), output / "apply_id.py")
    print("Updated features: " + ", ".join(sorted({item["variable"] for item in changes})))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Original ID's matching YAML")
    parser.add_argument("--id-correctionlib", required=True, help="Original full ID JSON or JSON.gz")
    parser.add_argument("--correction-lib", required=True, help="Original ID's matching calibration JSON")
    parser.add_argument("--candidate-config", required=True, help="Refit's candidate_id.yaml with the new calibration")
    parser.add_argument("--outputdir", required=True, help="New ID bundle directory")
    args = parser.parse_args()
    print(recalibrate_id(args.config, args.id_correctionlib, args.correction_lib,
                         args.candidate_config, args.outputdir))


if __name__ == "__main__":
    main()
