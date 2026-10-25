"""Validation-only mapped references with explicit inputs, formulas and provenance."""
import hashlib
from pathlib import Path

import numpy as np
import yaml

from workflow.InspectDistribution import define_variables
from workflow.working_points import working_points_from_spec


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
