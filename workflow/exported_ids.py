"""Validate exported IDs and remove their own cut components without training files."""
from copy import deepcopy
import gzip
import json
import re
import sys

import correctionlib
import numpy as np

from apply_id import evaluate_id, id_evaluators, id_inputs
from workflow.IDCorrection import id_expression_compiler
from workflow.working_points import WP


def read_json(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as handle:
        return json.load(handle)


def _expanded_formula(node, symbols):
    """Expand local TFormula coordinates and transforms; never evaluate source text."""
    if node["nodetype"] == "transform":
        updated = dict(symbols)
        updated[node["input"]] = _expanded_formula(node["rule"], symbols)
        return _expanded_formula(node["content"], updated)
    if node["nodetype"] != "formula" or node["parser"] != "TFormula":
        raise ValueError("N-minus-one labeling requires exported TFormula cut components")
    coordinates = dict(zip("xyzt", node["variables"]))
    return re.sub(r"\b[xyzt]\b", lambda match: symbols[coordinates[match.group()]], node["expression"])


def load_exported_wps(path, base, config, calibration_path, nminus1=False):
    """Match cut formulas to configured feature expressions, not training thresholds/order."""
    payload = read_json(path)
    corrections = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    evaluators = id_evaluators(corrections)
    compounds = {item["name"]: item for item in payload.get("compound_corrections", [])}
    components = {item["name"]: item for item in payload["corrections"]}
    features = {}
    symbols = {}
    if nminus1:
        compiler = id_expression_compiler(base, config, read_json(calibration_path))
        preparation = base["workflow"]["InspectDistribution"]
        candidates = dict.fromkeys([*preparation["input_columns"], *preparation.get("define_variable", {}),
                                   *config.get("validation", {}).get("define_variable", {})])
        features = {name: compiler.symbols[name].replace("e+", "e") for name in candidates}
        symbols = {name: f"__raw_{index}__" for index, name in enumerate(compiler.inputs)}
    sources = {}
    for source, options in config["wp_sources"].items():
        options = options if isinstance(options, dict) else {}
        names = [name for name in (compounds or components) if name in evaluators and name.startswith(source + "/")]
        if not names:
            raise ValueError(f"No exported WPs for source {source} in {path}")
        sources[source] = []
        for name in names:
            wp_name = name[len(source) + 1:]
            meta = dict(label=options.get("labels", {}).get(wp_name, wp_name),
                        reference_evaluator=name, reference_kind="Exported correctionlib ID")
            if nminus1:
                compound = compounds.get(name)
                if not compound or compound.get("inputs_update") or compound.get("output_op") != "*":
                    raise ValueError(f"{name}: N-minus-one needs independent multiplicative cut components")
                groups = {}
                roots = list(dict.fromkeys(re.match(re.escape(f"__id__/{name}/") + r"cut_\d+", item).group()
                                          for item in compound["stack"]
                                          if re.match(re.escape(f"__id__/{name}/") + r"cut_\d+(?:/|$)", item)))
                for root in roots:
                    expression = _expanded_formula(components[root]["data"], symbols)
                    matches = []
                    for variable, feature in features.items():
                        thresholds = re.findall(re.escape(feature) + r"(?:<=|>=|<|>)(-?\d+(?:\.\d*)?(?:e[+-]?\d+)?)", expression)
                        if any(abs(float(value)) < sys.float_info.max for value in thresholds):
                            matches.append(variable)
                    if len(matches) != 1:
                        raise ValueError(f"{root}: cannot identify the removed variable unambiguously ({matches}). "
                                         "Use the matching feature YAML and calibration for this export.")
                    groups.setdefault(matches[0], []).extend(item for item in compound["stack"]
                                                            if item == root or item.startswith(root + "/"))
                grouped = {item for group in groups.values() for item in group}
                common = [item for item in compound["stack"] if item not in grouped]
                if not groups or any(not re.fullmatch(r"__id__/(?:exclusive_regions|fiducial)(?:/finite_\d+)?", item)
                                     for item in common):
                    raise ValueError(f"{name}: unsupported or missing exported cut components")
                meta.update(id_components=groups, id_common_components=common)
            sources[source].append(WP(wp_name, {}, meta))
    return corrections, sources


def evaluate_exported_wps(frame, corrections, templates):
    """Keep full native decisions and auditable per-variable masks on the same rows."""
    sources = deepcopy(templates)
    columns, report = {}, []

    def component_column(name):
        if name not in columns:
            evaluator = corrections[name]
            values = np.asarray(evaluator.evaluate(*[
                frame[item.name].to_numpy(dtype=float) for item in id_inputs(evaluator)]))
            if not np.isin(values, [0., 1.]).all():
                raise ValueError(f"Nonbinary exported ID component: {name}")
            column = f"__id_component_{len(columns)}"
            if column in frame:
                raise ValueError(f"ID diagnostic column already exists: {column}")
            frame[column] = values == 1.
            columns[name] = column
        return columns[name]

    for wps in sources.values():
        for wp in wps:
            name = wp.meta["reference_evaluator"]
            passed = evaluate_id(corrections, name, frame)
            entry = dict(correction=name, rows=len(frame), correctionlib_pass=int(passed.sum()))
            if "id_components" in wp.meta:
                wp.meta["component_columns"] = {variable: [component_column(item) for item in group]
                                                for variable, group in wp.meta["id_components"].items()}
                common = [component_column(item) for item in wp.meta["id_common_components"]]
                valid_column = f"__id_valid_{len(report)}"
                frame[valid_column] = frame[common].all(axis=1).to_numpy() if common else True
                wp.meta["valid_column"] = valid_column
                all_columns = [item for group in wp.meta["component_columns"].values() for item in group]
                expected = frame[valid_column].to_numpy() & frame[all_columns].all(axis=1).to_numpy()
                entry.update(components=wp.meta["id_components"], common_components=wp.meta["id_common_components"],
                             mismatches=int((passed != expected).sum()))
                if entry["mismatches"]:
                    raise ValueError(f"Exported component decisions differ from full ID: {name}")
            column = f"__exported_id_pass_{len(report)}"
            frame[column] = passed
            wp.meta["pass_column"] = column
            report.append(entry)
    return sources, report
