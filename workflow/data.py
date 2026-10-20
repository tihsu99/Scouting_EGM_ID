"""Shared sample normalization, event splitting, and training weights."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def edges(spec):
    result = np.linspace(*spec[:2], int(spec[2]) + 1) if len(spec) == 3 else np.asarray(spec, dtype=float)
    if len(result) < 2 or not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise ValueError(f"Invalid histogram edges: {spec}")
    return result


def load_samples(inputdir, process, n_files=-1, normalization=None, columns=None):
    """Normalize selected files before candidate cuts; extensions share one denominator."""
    normalization = normalization or {}
    groups = normalization.get("groups", {})
    weight_column = normalization.get("weight_column", "Generator.weight")
    group_for = {}
    for group, cfg in groups.items():
        for sample in cfg["samples"]:
            if sample in group_for:
                raise ValueError(f"Sample {sample} belongs to multiple normalization groups")
            group_for[sample] = group
    names = sorted({s for cfg in process.values() for s in cfg["samples"]})
    frames, reports, totals = {}, {}, {}
    seen_inputs = set()
    for sample in names:
        paths = sorted((Path(inputdir) / sample).glob("*.parquet"))
        if n_files > 0:
            paths = paths[:n_files]
        if not paths:
            raise FileNotFoundError(f"No Parquet files for {sample} in {inputdir}")
        parts, metadata = [], []
        for path in paths:
            frame = pd.read_parquet(path, columns=columns)
            frame["sample"] = sample
            parts.append(frame)
            if normalization:
                receipt = json.loads(Path(str(path) + ".json").read_text())
                if receipt["sample"] != sample:
                    raise ValueError(f"Receipt sample mismatch: {path}")
                for source in receipt["input_files"]:
                    if source in seen_inputs:
                        raise ValueError(f"Duplicate input file: {source}")
                    seen_inputs.add(source)
                info = receipt.get("normalization")
                if not isinstance(info, dict):
                    raise ValueError(f"Missing preselection normalization: {path}. "
                                     "Recover metadata from the source events before weighted analysis; "
                                     "merged electron rows alone cannot supply it.")
                metadata.append(info)
        frames[sample] = pd.concat(parts, ignore_index=True)
        reports[sample] = {"files": [str(p) for p in paths], "rows": len(frames[sample])}
        if normalization:
            if sample not in group_for:
                raise ValueError(f"Missing normalization group for {sample}")
            group = group_for[sample]
            cfg = groups[group]
            total = sum(m["sum_genweight"] for m in metadata)
            totals[group] = totals.get(group, 0.0) + total
            reports[sample].update(group=group, sum_genweight=total,
                                   events=sum(m["events"] for m in metadata),
                                   sum_genweight2=sum(m.get("sum_genweight2", 0.) for m in metadata),
                                   negative_events=sum(m["negative_events"] for m in metadata))
            if "generator_range" in cfg:
                lo, hi = cfg["generator_range"]
                for m in metadata:
                    if m["generator_bin_min"] < lo or (hi is not None and m["generator_bin_max"] >= hi):
                        raise ValueError(f"Generator bin outside {group} range; verify dataset and binvar")
    # Only generation bins within the same family are required to be contiguous.
    families = {}
    for group in totals:
        cfg = groups[group]
        if cfg.get("stitch_family"):
            families.setdefault(cfg["stitch_family"], []).append(cfg["generator_range"])
    for family, ranges in families.items():
        ranges = sorted(ranges, key=lambda r: r[0])
        for left, right in zip(ranges, ranges[1:]):
            if left[1] is None or left[1] != right[0]:
                raise ValueError(f"Gap or overlap in {family}: {left}, {right}")
    for sample, frame in frames.items():
        if normalization:
            group = group_for[sample]
            cfg = groups[group]
            sigma = cfg.get("cross_section_pb")
            if sigma is None or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError(f"Set a verified cross_section_pb for normalization group {group}")
            if not np.isfinite(totals[group]) or totals[group] <= 0:
                raise ValueError(f"Nonpositive generator weight sum for {group}")
            factor = float(sigma) * normalization.get("luminosity_pb", 1.0) / totals[group]
            if not np.isfinite(factor) or factor <= 0:
                raise ValueError(f"Invalid normalization for {group}")
            frame["weight"] = frame[weight_column].to_numpy(dtype=float) * factor
            reports[sample].update(normalization_factor=factor, cross_section_pb=float(sigma),
                                   group_sum_genweight=totals[group],
                                   luminosity_pb=normalization.get("luminosity_pb", 1.0))
        else:
            frame["weight"] = 1.0
        if not np.isfinite(frame["weight"]).all():
            raise ValueError(f"Non-finite weights in {sample}")
    selected = {}
    for category, cfg in process.items():
        parts = []
        for sample, scale in cfg["samples"].items():
            frame = frames[sample].copy()
            frame["weight"] *= float(scale)
            if cfg.get("requirement"):
                frame = frame.query(cfg["requirement"]).copy()
            parts.append(frame)
        selected[category] = pd.concat(parts, ignore_index=True)
        if selected[category].empty:
            raise ValueError(f"No selected candidates for {category}")
    return selected, reports


def event_split(frame, fraction=0.8, seed=42, columns=None):
    columns = columns or ["sample", "run", "luminosityBlock", "event"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Regenerate ntuples with event identifiers; missing {sorted(missing)}")
    if not 0 < fraction < 1:
        raise ValueError("train_ratio must be between zero and one")
    keys = frame[columns].copy()
    keys["split_seed"] = seed
    hashed = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    mask = hashed.astype(np.float64) / float(2**64) < fraction
    return frame.loc[mask].copy(), frame.loc[~mask].copy()


def training_weights(frame, mode):
    weights = frame["weight"].to_numpy(dtype=float)
    if mode == "absolute":
        return np.abs(weights)
    if mode != "nonnegative" or (weights < 0).any():
        raise ValueError("Training needs nonnegative weights; explicitly choose absolute for signed MC")
    return weights.copy()


def reweight_training(signal, background, target, binning, outdir):
    """Fit on training data only; zero unsupported cells and report lost coverage."""
    if not target or str(target).lower() == "none":
        return
    target = str(target).lower()
    if target not in ("signal", "background"):
        raise ValueError("kinematic_reweighting must be SIGNAL, BACKGROUND, or none")
    variables = list(binning)
    bins = [edges(binning[v]) for v in variables]
    frames = {"signal": signal, "background": background}
    source = "signal" if target == "background" else "background"
    histograms = {name: np.histogramdd(df[variables].to_numpy(), bins=bins,
                                      weights=df.training_weight)[0] for name, df in frames.items()}
    hs, ht = histograms[source], histograms[target]
    if hs.sum() <= 0 or ht.sum() <= 0:
        raise ValueError("Empty training support in reweighting range")
    ratio = np.divide(ht / ht.sum(), hs / hs.sum(), out=np.zeros_like(hs), where=hs > 0)
    df = frames[source]
    indices, valid = [], np.ones(len(df), dtype=bool)
    for var, bins_v in zip(variables, bins):
        values = df[var].to_numpy()
        index = np.searchsorted(bins_v, values, side="right") - 1
        index[values == bins_v[-1]] = len(bins_v) - 2
        valid &= np.isfinite(values) & (index >= 0) & (index < len(bins_v) - 1)
        indices.append(np.clip(index, 0, len(bins_v) - 2))
    factors = np.where(valid, ratio[tuple(indices)], 0.0)
    original = df.training_weight.sum()
    df["training_weight"] *= factors
    if df.training_weight.sum() <= 0:
        raise ValueError("No common signal/background training support")
    df["training_weight"] *= original / df.training_weight.sum()
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    after = np.histogramdd(df[variables].to_numpy(), bins=bins, weights=df.training_weight)[0]
    report = {"source": source, "target": target, "variables": variables,
              "edges": [b.tolist() for b in bins], "ratio": ratio.tolist(),
              "source_before": hs.tolist(), "target_histogram": ht.tolist(), "source_after": after.tolist(),
              "target_weight_without_source_support": float(ht[hs <= 0].sum() / ht.sum()),
              "source_rows_outside_range": int((~valid).sum())}
    (path / "reweighting.json").write_text(json.dumps(report, indent=2) + "\n")


def efficiency_stats(passed, weights):
    """Delta-method variance for a weighted subset/total ratio, including signed MC."""
    passed, weights = np.asarray(passed, dtype=bool), np.asarray(weights, dtype=float)
    total, sumw2 = weights.sum(), np.square(weights).sum()
    if total <= 0:
        return float("nan"), float("nan"), 0.0
    eff = weights[passed].sum() / total
    variance = np.sum(np.square(weights) * np.square(passed.astype(float) - eff)) / total**2
    return float(eff), float(np.sqrt(variance)), float(total**2 / sumw2) if sumw2 else 0.0
