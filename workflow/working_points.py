"""Shared working-point loading, fixed cuts, and Markdown tables."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml


ID_USAGE_MARKDOWN = r"""## Using the exported ID

Use the matching `scouting_id.json.gz` and `apply_id.py` from this release.
Run the examples from the directory containing these two files. The helper is
standalone; the training repository, YAML files and separate calibration JSON
are not needed. Keep different releases in separate directories.

The JSON uses correctionlib schema v2. Complete WPs are in
`CorrectionSet.compound`; `__id__/...` entries are internal components.
Native evaluation returns `1.0` for pass and `0.0` for fail. `evaluate_id`
returns a Boolean array, with one decision per candidate.

Provide raw candidate columns with their original names. Use GeV for pT and
absolute isolation, and the event rho for each candidate. Do not pre-apply EA,
pT/scale corrections, isolation division or clipping: the JSON already applies
the configured feature definitions, region selections and WP cuts. Only the
inputs listed by each WP are required; truth labels and event weights are not
inputs to this ID decision. Extra DataFrame columns are allowed.

Install the dependencies if needed:

```bash
python3 -m pip install correctionlib numpy pandas pyarrow
```

### Python: inspect inputs and evaluate all WPs

Replace `candidates.parquet` with a candidate-level Parquet containing the raw
columns. WP names and input order are read from the supplied JSON, so the same
example works with either release.

```python
import correctionlib
import pandas as pd

from apply_id import evaluate_id, id_evaluators, id_inputs

ids = correctionlib.CorrectionSet.from_file("scouting_id.json.gz")
evaluators = id_evaluators(ids)
names = list(evaluators)

for name, wp in evaluators.items():
    print(name, [(item.name, item.type) for item in id_inputs(wp)])

candidates = pd.read_parquet("candidates.parquet")
# To apply one WP, set names to a list containing its printed name.
for name in names:
    candidates[f"pass_{name}"] = evaluate_id(ids, name, candidates)

print(candidates[[f"pass_{name}" for name in names]].head())
```

### Command line: stream a large Parquet and save decisions

This retains all input columns and adds Boolean `pass_<source>/<WP>` columns.
The output must be a new file. Add `--wps` followed by names printed above to
select particular WPs; omitting it evaluates all WPs in the JSON.

```bash
python3 apply_id.py scouting_id.json.gz \
  --input candidates.parquet \
  --output candidates_with_id.parquet \
  --batch-size 200000
```
"""


def load_id_config(path):
    """Read an existing validation recipe or a standalone pipeline's evaluation section."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if "base_config" in config:
        base_path = (path.parent / config["base_config"]).resolve()
        specification = yaml.safe_load(base_path.read_text())
        if "evaluation" in specification:
            evaluation, base, _ = load_id_config(base_path)
            for source, options in evaluation.get("wp_sources", {}).items():
                if isinstance(options, str):
                    evaluation["wp_sources"][source] = options.replace("{config_dir}", str(base_path.parent))
                else:
                    options["path"] = options["path"].replace("{config_dir}", str(base_path.parent))
            for source, template in evaluation.get("validation", {}).get("reference_sources", {}).items():
                evaluation["validation"]["reference_sources"][source] = template.replace("{config_dir}", str(base_path.parent))
            for options in evaluation.get("validation", {}).get("correctionlib_references", {}).values():
                options["path"] = options["path"].replace("{config_dir}", str(base_path.parent))
            # Profiles inherit the pipeline's physics definitions; explicit empty
            # reference dictionaries replace inherited references.
            for key, value in config.items():
                if key in ("validation", "plot_style"):
                    evaluation[key] = dict(evaluation.get(key, {}), **value)
                elif key != "base_config":
                    evaluation[key] = value
            return evaluation, base, base_path
        return config, specification, base_path
    if "workflow" not in config or "evaluation" not in config:
        raise ValueError("ID configuration needs base_config or an inline workflow with evaluation")
    base = deepcopy(config.get("shared", {}))
    base["workflow"] = deepcopy(config["workflow"])
    for module, overrides in config.get("workflow_overrides", {}).items():
        base["workflow"][module].update(overrides)
    preparation = base["workflow"]["InspectDistribution"]
    base["workflow"]["InspectDistribution"] = dict(config.get("shared", {}), **preparation)
    evaluation = deepcopy(config["evaluation"])
    validation = deepcopy(base["workflow"]["ValidateID"])
    evaluation.setdefault("regions", validation.pop("regions"))
    validation.pop("label", None)
    validation.update(evaluation.get("validation", {}))
    evaluation["validation"] = validation
    evaluation["plot_style"] = dict(base.get("plot_style", {}), **evaluation.get("plot_style", {}))
    return evaluation, base, path


@dataclass
class WP:
    name: str
    window_cuts: Dict[str, Tuple[float, float]]
    meta: dict


def normalize_fixed_cuts(cuts):
    result = {}
    for variable, bounds in (cuts or {}).items():
        if not bounds or set(bounds) - {"min", "max"}:
            raise ValueError(f"Fixed cut {variable} requires min and/or max")
        values = {key: float(value) for key, value in bounds.items()}
        if not np.isfinite(list(values.values())).all() or values.get("min", -np.inf) > values.get("max", np.inf):
            raise ValueError(f"Invalid fixed cut: {variable} {bounds}")
        result[variable] = values
    return result


def pass_fixed_cuts(frame, cuts):
    """Fixed bounds are inclusive, independent of the optimized cuts' strictness."""
    passed = np.ones(len(frame), dtype=bool)
    for variable, bounds in cuts.items():
        values = frame[variable].to_numpy()
        passed &= np.isfinite(values)
        if "min" in bounds:
            passed &= values >= bounds["min"]
        if "max" in bounds:
            passed &= values <= bounds["max"]
    return passed


def load_tmva_wp_file(path):
    with open(path) as handle:
        specification = json.load(handle)
    return working_points_from_spec(specification)


def working_points_from_spec(specification):
    """Read the shared cut format from a JSON file or an inline reference recipe."""
    wps = []
    for name, entry in specification.items():
        if "window_cuts" in entry:
            windows = {key: tuple(map(float, bounds)) for key, bounds in entry["window_cuts"].items()}
        elif "cuts" in entry:
            windows = {key: (-1e30, float(value)) for key, value in entry["cuts"].items()}
        else:
            raise KeyError("No 'window_cuts' or 'cuts' in WP JSON entry")
        meta = {key: value for key, value in entry.items() if key not in ("cuts", "window_cuts")}
        meta["fixed_cuts"] = normalize_fixed_cuts(meta.get("fixed_cuts"))
        wps.append(WP(name, windows, meta))
    wps.sort(key=lambda wp: wp.meta.get("eff_selected_from_tmva", np.nan))
    return wps


def configured_wp_sources(base, validation):
    """Use the same fixed-cut recipe for training, existing WPs, and exports."""
    recipes = base["workflow"].get("TrainCutBasedID_TMVA", {}).get("fixed_cuts", {})
    sources = deepcopy(validation["wp_sources"])
    for source, options in sources.items():
        if isinstance(options, str):
            options = sources[source] = {"path": options}
        options.setdefault("fixed_cuts", deepcopy(recipes.get(source, {})))
    return sources


def load_configured_wps(path, options=None):
    options = options or {}
    wps = load_tmva_wp_file(path)
    for wp in wps:
        expected = options.get("isolation_denominator")
        if expected is not None and wp.meta.get("isolation_denominator") != expected:
            raise ValueError(f"Isolation denominator mismatch in {path}/{wp.name}: expected {expected}")
        mapping = options.get("variable_map", {})
        wp.window_cuts = {mapping.get(var, var): cut for var, cut in wp.window_cuts.items()}
        fixed = dict(wp.meta["fixed_cuts"])
        fixed.update(normalize_fixed_cuts(options.get("fixed_cuts", {}).get(wp.name)))
        wp.meta["fixed_cuts"] = {mapping.get(var, var): cut for var, cut in fixed.items()}
        wp.meta["label"] = options.get("labels", {}).get(wp.name, wp.name)
    if not wps:
        raise ValueError(f"No working points in {path}")
    return wps


def cut_bounds(wp, variable):
    """Displayable bounds, preserving strict optimized and inclusive fixed cuts."""
    bounds = []
    if variable in wp.window_cuts:
        low, high = wp.window_cuts[variable]
        strict = wp.meta.get("strict", False)
        if low > -1e30:
            bounds.append((">" if strict else "≥", low))
        if high < 1e30:
            bounds.append(("<" if strict else "≤", high))
    bounds.extend(("≥" if key == "min" else "≤", value)
                  for key, value in wp.meta.get("fixed_cuts", {}).get(variable, {}).items())
    return bounds


def calibration_table_lines(definitions, calibration, precision):
    """Show configured coefficient clipping while retaining original eta binning."""
    lines = []
    if definitions:
        lines.extend(["## Feature definitions", "",
                      "Apply these definitions in the listed order before the WP cuts. "
                      "The expressions retain the configured pT dependence and clipping.", "",
                      "| Variable | Expression |", "|---|---|"])
        for name, definition in definitions.items():
            expression = definition if isinstance(definition, str) else definition["expression"]
            if isinstance(definition, dict):
                for bound, function in (("lower", "max"), ("upper", "min")):
                    if definition.get(bound) is not None:
                        expression = f"{function}({definition[bound]:.{precision}g}, {expression})"
            expression = expression.replace("|", r"\|")
            lines.append(f"| `{name}` | ``{expression}`` |")
        lines.append("")
    if calibration is None:
        if definitions:
            lines.extend(["Coefficient values are omitted; supply --correction-lib to print them.", ""])
        return lines

    from workflow.IDCorrection import coefficient_expression

    groups = {}
    for correction in calibration["corrections"]:
        # Share the exporter's validation of scalar and clamped absolute-eta coefficients.
        coefficient_expression(correction, "eta")
        data = correction["data"]
        edges = tuple(data["edges"]) if isinstance(data, dict) else ()
        values = data["content"] if isinstance(data, dict) else [data]
        definition = (definitions or {}).get(correction["name"])
        if isinstance(definition, dict) and definition["expression"].strip() == correction["name"]:
            for bound, function in (("lower", max), ("upper", min)):
                if definition.get(bound) is not None:
                    values = [function(definition[bound], value) for value in values]
        groups.setdefault(edges, {})[correction["name"]] = values
    lines.extend(["## Calibration coefficients", "",
                  "Configured coefficient lower/upper bounds are already applied to the displayed values. "
                  "Coefficients are evaluated at abs(eta). Tables group only identical eta bins; "
                  "no coefficients are rebinned or interpolated. Binned coefficients use flow=clamp "
                  "outside the tabulated edges. The ID region selections above still apply.", ""])
    for index, (edges, coefficients) in enumerate(groups.items(), 1):
        lines.extend([f"### Coefficient table {index}", "",
                      "| `abs(eta)` interval | " + " | ".join(f"`{name}`" for name in coefficients) + " |",
                      "|---|" + "---|" * len(coefficients)])
        for row in range(len(edges) - 1 if edges else 1):
            interval = f"[{edges[row]:.{precision}g}, {edges[row + 1]:.{precision}g})" if edges else "All"
            values = " | ".join(f"{values[row]:.{precision}g}" for values in coefficients.values())
            lines.append(f"| {interval} | {values} |")
        lines.append("")
    return lines


def print_id_table(wps_by_region, output=None, requirement=None, region_definitions=None, precision=6,
                   definitions=None, calibration=None, include_usage=False):
    """Print a Markdown cut table, optionally save it, and return its contents."""
    lines = ["# Cut-based ID", ""]
    if requirement:
        lines.extend([f"Common selection: `{requirement}`.", ""])
    for region, sources in wps_by_region.items():
        lines.extend([f"## {region}", ""])
        if region_definitions and region in region_definitions:
            lines.extend([f"Region selection: `{region_definitions[region]['requirement']}`.", ""])
        for source, wps in sources.items():
            lines.extend([f"### {source}", ""])
            if wps and wps[0].meta.get("reference_kind"):
                lines.extend([wps[0].meta["reference_kind"] + ". Definitions and source hashes: `reference_ids.json`.", ""])
                lines.extend(["- " + note for note in wps[0].meta.get("reference_limitations", [])])
                lines.append("")
            lines.extend(["| Variable | " + " | ".join(wp.meta.get("label", wp.name) for wp in wps) + " |",
                          "|---|" + "---|" * len(wps)])
            if any(wp.meta.get("reference_evaluator") for wp in wps):
                lines.append("| Stored evaluator | " + " | ".join(
                    "`" + wp.meta.get("reference_evaluator", "") + "`" for wp in wps) + " |")
            variables = dict.fromkeys(variable for wp in wps
                                      for variable in [*wp.window_cuts, *wp.meta.get("fixed_cuts", {})])
            for variable in variables:
                entries = []
                for wp in wps:
                    terms = [f"{operator} {value:.{precision}g}" for operator, value in cut_bounds(wp, variable)]
                    entries.append(" and ".join(terms) or "No cut")
                lines.append(f"| `{variable}` | " + " | ".join(entries) + " |")
            lines.append("")
    lines.extend(calibration_table_lines(definitions, calibration, precision))
    lines.extend(["All cuts in a column are applied together, after that ID's configured feature calibration. "
                  "Stored correctionlib evaluators run directly on raw inputs.",
                  "Fixed cuts use inclusive bounds. Optimized WP names denote nominal optimization targets; "
                  "reference names retain their source definitions. Efficiencies after all cuts must be measured.",
                  f"Numbers are displayed to {precision} significant digits; JSON retains the exact thresholds and coefficients.", ""])
    if include_usage:
        lines.extend(ID_USAGE_MARKDOWN.splitlines())
        lines.append("")
    markdown = "\n".join(lines)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown)
    print(markdown)
    return markdown
