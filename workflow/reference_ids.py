"""Validation-only mapped references with explicit inputs, formulas and provenance."""
import hashlib
from pathlib import Path

import numpy as np
import yaml

from workflow.InspectDistribution import define_variables
from workflow.working_points import WP, working_points_from_spec


def load_correctionlib_references(sources, config_dir="."):
    """Load independent raw-input ID evaluators, without applying current calibration."""
    import correctionlib
    from apply_id import id_evaluators, id_inputs

    references = {}
    for source, options in (sources or {}).items():
        path = Path(options["path"].format(config_dir=config_dir)).resolve()
        corrections = correctionlib.CorrectionSet.from_file(str(path))
        evaluators = id_evaluators(corrections)
        prefix = options.get("prefix", "")
        selected = {name[len(prefix):]: name for name in sorted(evaluators) if name.startswith(prefix)}
        if not selected or any(not name for name in selected):
            raise ValueError(f"No ID working points with prefix {prefix!r} in {path}; available: {list(evaluators)}")
        references[source] = dict(
            corrections=corrections,
            metadata=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          working_points=selected, labels=options.get("labels", {}),
                          input_columns=sorted({item.name for name in selected.values()
                                                for item in id_inputs(evaluators[name])}),
                          evaluation="raw inputs with the reference artifact's own calibration"))
    return references


def add_correctionlib_reference(frame, source, reference):
    from apply_id import evaluate_id

    metadata = reference["metadata"]
    missing = set(metadata["input_columns"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing inputs for reference {source}: {sorted(missing)}")
    wps = []
    for index, (name, evaluator) in enumerate(metadata["working_points"].items()):
        column = f"__reference_pass_{source}_{index}"
        if column in frame:
            raise ValueError(f"Reference {source} would overwrite column {column}")
        frame[column] = evaluate_id(reference["corrections"], evaluator, frame)
        wps.append(WP(name, {}, dict(pass_column=column, fixed_cuts={},
                                    label=metadata["labels"].get(name, name),
                                    reference_evaluator=evaluator,
                                    reference_kind="independent correctionlib ID")))
    return wps


def load_reference_recipes(sources, config_dir="."):
    recipes = {}
    for source, template in (sources or {}).items():
        path = Path(template.format(config_dir=config_dir)).resolve()
        payload = path.read_bytes()
        recipe = yaml.safe_load(payload)
        for key in ("provenance", "mapping", "limitations", "input_columns", "regions"):
            if not recipe.get(key):
                raise ValueError(f"Reference {source} is missing {key}: {path}")
        recipes[source] = dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), recipe=recipe)
    return recipes


def add_reference_features(frame, source, recipe, region):
    """Evaluate on the existing denominator; invalid reference inputs fail only that ID."""
    columns = recipe["input_columns"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing inputs for reference {source}: {sorted(missing)}")
    regional = recipe["regions"][region]
    definitions = dict(recipe.get("define_variable", {}), **regional.get("define_variable", {}))
    tables = recipe.get("lookup_tables", {})
    valid_column = f"__reference_valid_{source}"
    added = set(tables) | set(definitions) | {valid_column}
    if added & set(frame.columns):
        raise ValueError(f"Reference {source} would overwrite columns: {sorted(added & set(frame.columns))}")
    valid = np.ones(len(frame), dtype=bool)
    for column in columns:
        valid &= np.isfinite(frame[column].to_numpy(dtype=float))
    for name, table in tables.items():
        edges = np.asarray(table["edges"], dtype=float)
        values = np.asarray(table["values"], dtype=float)
        if (edges.ndim != 1 or values.ndim != 1 or len(edges) != len(values) + 1
                or not len(values) or not np.isfinite(edges).all()
                or not np.isfinite(values).all() or not (np.diff(edges) > 0).all()):
            raise ValueError(f"Invalid reference lookup table {source}/{name}")
        coordinates = frame.eval(table["expression"]).to_numpy(dtype=float)
        indices = np.searchsorted(edges, coordinates, side="right") - 1
        inside = np.isfinite(coordinates) & (coordinates >= edges[0]) & (coordinates < edges[-1])
        # Outside the reference's tabulated domain, fail the ID without dropping candidates.
        frame[name] = np.where(inside, values[np.clip(indices, 0, len(values) - 1)], np.nan)
        valid &= inside
    define_variables(frame, definitions)
    for requirement in (recipe.get("requirement"), regional.get("requirement")):
        if requirement:
            valid &= frame.eval(requirement).to_numpy(dtype=bool)
    for column in definitions:
        valid &= np.isfinite(frame[column].to_numpy(dtype=float))
    frame[valid_column] = valid
    wps = working_points_from_spec(regional["working_points"])
    if not wps:
        raise ValueError(f"No working points for reference {source}/{region}")
    for wp in wps:
        wp.meta["valid_column"] = valid_column
        wp.meta["reference_kind"] = recipe.get("kind", "mapped reference")
        wp.meta["reference_limitations"] = recipe["limitations"]
    return wps
