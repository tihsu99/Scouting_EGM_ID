"""Shared working-point loading, fixed cuts, and Markdown tables."""
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


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


def print_id_table(wps_by_region, output=None, requirement=None, region_definitions=None, precision=6):
    """Print a Markdown cut table, optionally save it, and return its contents."""
    lines = ["# Cut-based electron ID", ""]
    if requirement:
        lines.extend([f"Common selection: `{requirement}`.", ""])
    for region, sources in wps_by_region.items():
        lines.extend([f"## {region}", ""])
        if region_definitions and region in region_definitions:
            lines.extend([f"Region selection: `{region_definitions[region]['requirement']}`.", ""])
        for source, wps in sources.items():
            lines.extend([f"### {source}", "",
                          "| Variable | " + " | ".join(wp.meta.get("label", wp.name) for wp in wps) + " |",
                          "|---|" + "---|" * len(wps)])
            variables = dict.fromkeys(variable for wp in wps
                                      for variable in [*wp.window_cuts, *wp.meta.get("fixed_cuts", {})])
            for variable in variables:
                entries = []
                for wp in wps:
                    terms = []
                    if variable in wp.window_cuts:
                        low, high = wp.window_cuts[variable]
                        strict = wp.meta.get("strict", False)
                        if low > -1e30:
                            terms.append(f"{'>' if strict else '≥'} {low:.{precision}g}")
                        if high < 1e30:
                            terms.append(f"{'<' if strict else '≤'} {high:.{precision}g}")
                    for bound, value in wp.meta.get("fixed_cuts", {}).get(variable, {}).items():
                        terms.append(f"{'≥' if bound == 'min' else '≤'} {value:.{precision}g}")
                    entries.append(" and ".join(terms) or "No cut")
                lines.append(f"| `{variable}` | " + " | ".join(entries) + " |")
            lines.append("")
    lines.extend(["All cuts in a column are applied together, after the configured feature calibration.",
                  "Fixed cuts use inclusive bounds. WP names denote nominal optimization targets; "
                  "efficiencies after fixed cuts must be measured.",
                  f"Numbers are displayed to {precision} significant digits; JSON retains the exact thresholds.", ""])
    markdown = "\n".join(lines)
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown)
    print(markdown)
    return markdown
