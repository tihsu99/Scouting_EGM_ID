"""Shared sample normalization, event splitting, and training weights."""
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from workflow.sampling import load_random_sample


def edges(spec):
    result = np.linspace(*spec[:2], int(spec[2]) + 1) if len(spec) == 3 else np.asarray(spec, dtype=float)
    if len(result) < 2 or not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise ValueError(f"Invalid histogram edges: {spec}")
    return result


def sample_normalizations(sample_paths, normalization):
    """Resolve event-level receipt denominators without reading candidate rows."""
    groups = normalization.get("groups", {})
    group_for = {}
    for group, cfg in groups.items():
        for sample in cfg["samples"]:
            if sample in group_for:
                raise ValueError(f"Sample {sample} belongs to multiple normalization groups")
            group_for[sample] = group
    reports, totals, seen_inputs = {}, {}, set()
    for sample, paths in sample_paths.items():
        metadata = []
        for path in paths:
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
                                 "merged candidate rows alone cannot supply it.")
            metadata.append(info)
        if sample not in group_for:
            raise ValueError(f"Missing normalization group for {sample}")
        group = group_for[sample]
        cfg = groups[group]
        total = sum(m["sum_genweight"] for m in metadata)
        totals[group] = totals.get(group, 0.0) + total
        reports[sample] = dict(group=group, sum_genweight=total,
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
    for sample in sample_paths:
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
        reports[sample].update(normalization_factor=factor, cross_section_pb=float(sigma),
                               group_sum_genweight=totals[group],
                               luminosity_pb=normalization.get("luminosity_pb", 1.0))
    return reports


def load_samples(inputdir, process, n_files=-1, normalization=None, columns=None,
                 max_rows_per_sample=None, batch_size=100000, progress=False, sampling=None):
    """Load samples with a prefix cap or explicitly requested, rescaled random sampling."""
    if max_rows_per_sample is not None and (
            isinstance(max_rows_per_sample, bool) or not isinstance(max_rows_per_sample, int)
            or (max_rows_per_sample < 1 and max_rows_per_sample != -1)):
        raise ValueError("max_rows_per_sample must be a positive integer, null, or -1")
    limit = None if max_rows_per_sample == -1 else max_rows_per_sample
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    normalization = normalization or {}
    if sampling is not None:
        if not isinstance(sampling, Mapping) or sampling.get("method") != "random":
            raise ValueError("sampling.method must be random")
        if set(sampling) - {"method", "seed", "cache_dir"}:
            raise ValueError("Unknown sampling options; use method, seed, cache_dir")
    # A prefix has no known random inclusion probability to correct physical yields.
    if limit is not None and normalization and sampling is None:
        raise ValueError("max_rows_per_sample cannot be combined with cross-section normalization")
    weight_column = normalization.get("weight_column", "Generator.weight")
    names = sorted({s for cfg in process.values() for s in cfg["samples"]})
    frames, reports, sample_paths = {}, {}, {}
    for sample in names:
        paths = sorted((Path(inputdir) / sample).glob("*.parquet"))
        if n_files > 0:
            paths = paths[:n_files]
        if not paths:
            raise FileNotFoundError(f"No Parquet files for {sample} in {inputdir}")
        sample_paths[sample] = paths
    normalized = sample_normalizations(sample_paths, normalization) if normalization else {}
    for sample, paths in sample_paths.items():
        if sampling is not None:
            frames[sample], reports[sample] = load_random_sample(
                paths, columns, limit, sampling.get("seed", 42), sample,
                sampling.get("cache_dir"), batch_size, progress)
            frames[sample]["sample"] = sample
            continue
        parts, read_paths = [], []
        rows_read = 0
        for path in paths:
            if limit is not None and rows_read >= limit:
                break
            if limit is None and not progress:
                parts.append(pd.read_parquet(path, columns=columns))
            else:
                with path.open("rb") as source:
                    parquet = pq.ParquetFile(source)
                    missing = set(columns or []) - set(parquet.schema_arrow.names)
                    if missing:
                        raise ValueError(f"Missing requested Parquet columns in {path}: {sorted(missing)}")
                    remaining = None if limit is None else limit - rows_read
                    total = parquet.metadata.num_rows
                    if remaining is not None:
                        total = min(total, remaining)
                    read_batch_size = batch_size if remaining is None else min(batch_size, remaining)
                    with tqdm(total=total, desc=f"Read {sample}/{path.name}", unit="row",
                              unit_scale=True, dynamic_ncols=True, disable=not progress) as bar:
                        for batch in parquet.iter_batches(batch_size=read_batch_size,
                                                          columns=columns, use_threads=False):
                            if limit is not None:
                                batch = batch.slice(0, min(batch.num_rows, limit - rows_read))
                            parts.append(batch.to_pandas(use_threads=False))
                            rows_read += batch.num_rows
                            bar.update(batch.num_rows)
                            if limit is not None and rows_read >= limit:
                                break
            read_paths.append(path)
        if not parts:
            raise ValueError(f"No candidates in {sample}")
        frames[sample] = pd.concat(parts, ignore_index=True)
        del parts  # Do not retain all input batches alongside the concatenated table.
        frames[sample]["sample"] = sample
        # Preserve the input position before cuts; keep the same sorted files when reusing a split.
        frames[sample]["sample_row"] = np.arange(len(frames[sample]), dtype=np.uint64)
        reports[sample] = {"files": [str(p) for p in read_paths], "rows": len(frames[sample])}
    for sample, frame in frames.items():
        if normalization:
            reports[sample].update(normalized[sample])
            frame["weight"] = frame[weight_column].to_numpy(dtype=float) * normalized[sample]["normalization_factor"]
        else:
            frame["weight"] = 1.0
        frame["weight"] *= reports[sample].get("sampling_rescale", 1.0)
        if not np.isfinite(frame["weight"]).all():
            raise ValueError(f"Non-finite weights in {sample}")
    selected = {}
    for category, cfg in process.items():
        parts = []
        for sample, scale in cfg["samples"].items():
            # Replace only the weight column before cuts; do not copy rejected rows.
            frame = frames[sample].copy(deep=False)
            frame["weight"] = frame["weight"] * float(scale)
            if cfg.get("requirement"):
                frame = frame.query(cfg["requirement"])
            else:
                frame = frame.copy()
            parts.append(frame)
        selected[category] = pd.concat(parts, ignore_index=True)
        if selected[category].empty:
            raise ValueError(f"No selected candidates for {category}")
    return selected, reports


def event_split(frame, fraction=0.8, seed=42, columns=None):
    # Luigi freezes YAML lists into tuples; pandas needs a list for multiple columns.
    columns = list(columns or ["sample", "run", "luminosityBlock", "event"])
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns {sorted(missing)}; configure split_columns for "
                         "identifiers present in the data")
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


def kinematic_weights(source, target, binning, weight_col="weight"):
    """Match source to target on shared histogram support, preserving source sumw."""
    variables = list(binning)
    if not variables:
        raise ValueError("Kinematic reweighting needs at least one variable")
    bins = [edges(binning[v]) for v in variables]
    histograms = []
    for frame in (source, target):
        weights = frame[weight_col].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Kinematic reweighting requires finite, nonnegative weights with positive total")
        histograms.append(np.histogramdd(frame[variables].to_numpy(), bins=bins, weights=weights)[0])
    hs, ht = histograms
    if hs.sum() <= 0 or ht.sum() <= 0:
        raise ValueError("Empty source/target support in reweighting range")
    ratio = np.divide(ht / ht.sum(), hs / hs.sum(), out=np.zeros_like(hs), where=hs > 0)
    indices, valid = [], np.ones(len(source), dtype=bool)
    for var, bins_v in zip(variables, bins):
        values = source[var].to_numpy()
        index = np.searchsorted(bins_v, values, side="right") - 1
        index[values == bins_v[-1]] = len(bins_v) - 2
        valid &= np.isfinite(values) & (index >= 0) & (index < len(bins_v) - 1)
        indices.append(np.clip(index, 0, len(bins_v) - 2))
    factors = np.where(valid, ratio[tuple(indices)], 0.0)
    weights = source[weight_col].to_numpy(dtype=float) * factors
    if weights.sum() <= 0:
        raise ValueError("No common source/target support for kinematic reweighting")
    scale = source[weight_col].sum() / weights.sum()
    weights *= scale
    after = np.histogramdd(source[variables].to_numpy(), bins=bins, weights=weights)[0]
    report = {"variables": variables, "edges": [b.tolist() for b in bins],
              "ratio": ratio.tolist(), "normalization": float(scale),
              "source_before": hs.tolist(), "target_histogram": ht.tolist(), "source_after": after.tolist(),
              "target_weight_without_source_support": float(ht[hs <= 0].sum() / ht.sum()),
              "target_weight_outside_range": float(max(0., 1. - ht.sum() / target[weight_col].sum())),
              "source_rows_outside_range": int((~valid).sum()),
              "source_rows_zero_weight": int((weights == 0).sum()),
              "source_effective_entries": float(weights.sum() ** 2 / np.square(weights).sum())}
    return weights, report


def reweight_training(signal, background, target, binning, outdir):
    """Fit on training data only; zero unsupported cells and report lost coverage."""
    if not target or str(target).lower() == "none":
        return
    target = str(target).lower()
    if target not in ("signal", "background"):
        raise ValueError("kinematic_reweighting must be SIGNAL, BACKGROUND, or none")
    frames = {"signal": signal, "background": background}
    source = "signal" if target == "background" else "background"
    weights, report = kinematic_weights(frames[source], frames[target], binning, "training_weight")
    frames[source]["training_weight"] = weights
    report.update(source=source, target=target)
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
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
