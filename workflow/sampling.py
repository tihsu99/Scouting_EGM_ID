"""Bounded candidate subsets with reusable raw-column caches."""
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import random
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def sample_training_candidates(frame, size, config, seed, weight_column):
    """Sample within sample/kinematic strata and correct for inclusion probability."""
    if not isinstance(config, Mapping) or set(config) != {"variable", "edges", "factors"}:
        raise ValueError("training_sampling requires variable, edges, factors")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("Training sample size must be a positive integer")
    variable = config["variable"]
    boundaries = np.asarray([np.inf if value is None else value for value in config["edges"]], dtype=float)
    factors = np.asarray(config["factors"], dtype=float)
    if (boundaries.ndim != 1 or len(boundaries) < 2 or not np.isfinite(boundaries[:-1]).all()
            or np.isnan(boundaries).any() or np.any(np.diff(boundaries) <= 0)
            or factors.shape != (len(boundaries) - 1,) or not np.isfinite(factors).all()
            or (factors <= 0).any()):
        raise ValueError("training_sampling needs increasing edges and one positive factor per bin")
    missing = {"sample", variable, weight_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing training sampling columns: {sorted(missing)}")
    weights = frame[weight_column].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Training sampling requires finite, nonnegative weights")
    # Zero-weight candidates cannot contribute to the optimizer's requested statistics.
    eligible_positions = np.flatnonzero(weights > 0)
    eligible = frame[["sample", variable]].iloc[eligible_positions]
    values = eligible[variable].to_numpy(dtype=float)
    bins = np.searchsorted(boundaries, values, side="right") - 1
    if (not np.isfinite(values).all() or eligible["sample"].isna().any()
            or (bins < 0).any() or (bins >= len(factors)).any()):
        raise ValueError("Training sampling bins must cover every positive-weight candidate")
    if len(eligible) < size:
        raise ValueError(f"Need {size:,} positive-weight prepared candidates, found {len(eligible):,}; "
                         "increase the prepared pool or reduce the configured training counts")
    groups = list(eligible.groupby([eligible["sample"], bins], sort=True, observed=True).indices.items())
    counts = np.asarray([len(positions) for _, positions in groups], dtype=np.int64)
    if size < len(groups):
        raise ValueError("Training sample size must cover every populated sample/bin stratum")
    priorities = counts * np.asarray([factors[bin_id] for (_, bin_id), _ in groups])
    if not np.isfinite(priorities.sum()):
        raise ValueError("Training sampling factors are too large")
    # Reserve one row per stratum, then redistribute quotas when small strata saturate.
    quotas = np.ones(len(groups), dtype=np.int64)
    while quotas.sum() < size:
        active = np.flatnonzero(quotas < counts)
        shares = (size - quotas.sum()) * priorities[active] / priorities[active].sum()
        additions = np.minimum(np.floor(shares).astype(np.int64), counts[active] - quotas[active])
        if not additions.any():
            order = np.argsort(-shares, kind="stable")[:int(size - quotas.sum())]
            additions[order] = 1
        quotas[active] += additions
    rng = np.random.default_rng(seed)
    positions, corrections, strata = [], [], []
    for ((sample, bin_id), population), count in zip(groups, quotas):
        chosen = rng.choice(population, size=int(count), replace=False)
        correction = len(population) / int(count)
        positions.append(chosen)
        corrections.append(np.full(int(count), correction))
        strata.append(dict(sample=str(sample), bin=int(bin_id), available=len(population),
                           selected=int(count), factor=float(factors[bin_id]),
                           inclusion_probability=int(count) / len(population),
                           weight_rescale=correction))
    selected = frame.iloc[eligible_positions[np.concatenate(positions)]].copy()
    selected[weight_column] = selected[weight_column].to_numpy(dtype=float) * np.concatenate(corrections)
    selected = selected.iloc[rng.permutation(len(selected))].reset_index(drop=True)
    report = dict(method="stratified_without_replacement", seed=seed,
                  variable=variable, edges=list(config["edges"]), factors=factors.tolist(),
                  input_rows=len(frame), zero_weight_rows=int((weights == 0).sum()),
                  available=len(eligible), selected=len(selected),
                  weight_column=weight_column, weight_correction="available / selected within each sample/bin",
                  input_sumw=float(weights.sum()), selected_sumw=float(selected[weight_column].sum()),
                  strata=strata)
    return selected, report


def load_sample_subset(paths, columns, limit, seed, sample, cache_dir, batch_size, progress,
                       method="random"):
    """Select before cuts; prefix rescaling assumes representative source ordering."""
    if limit is None:
        raise ValueError("Subset loading requires a positive max_rows_per_sample")
    if method not in ("random", "prefix"):
        raise ValueError("sampling.method must be random or prefix")
    if method == "random" and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError("sampling.seed must be a nonnegative integer")
    sources = []
    for path in paths:
        stat = path.stat()
        sources.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    identity = dict(version=1, sample=sample, sources=sources,
                    columns=None if columns is None else list(columns), limit=limit, seed=seed)
    if method == "prefix":
        identity.pop("seed")
        identity["method"] = method
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir).expanduser().resolve() / sample / f"{digest}.parquet" if cache_dir else None
    receipt = Path(str(cache) + ".json") if cache else None
    if cache and cache.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Sampling cache identity mismatch: {receipt}")
        frame = pd.read_parquet(cache)
        if len(frame) != saved["report"]["rows"] or "sample_row" not in frame:
            raise ValueError(f"Incomplete sampling cache: {cache}")
        report = dict(saved["report"], cache_hit=True)
        if progress:
            tqdm.write(f"Reuse {len(frame):,} cached candidates from {sample}")
        return frame, report

    file_rows, total = [], 0
    for path in paths:
        with path.open("rb") as source:
            parquet = pq.ParquetFile(source)
            missing = set(columns or []) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"Missing requested Parquet columns in {path}: {sorted(missing)}")
            file_rows.append(parquet.metadata.num_rows)
            total += parquet.metadata.num_rows
    if total == 0:
        raise ValueError(f"No candidates in {sample}")
    size = min(total, limit)
    if method == "prefix":
        indices = np.arange(size, dtype=np.int64)
    else:
        # range sampling avoids allocating a permutation of the full production.
        sample_seed = hashlib.sha256(f"{seed}:{sample}".encode()).hexdigest()
        indices = np.array(sorted(random.Random(sample_seed).sample(range(total), size)), dtype=np.int64)
    parts, offset, source_rows = [], 0, 0
    with tqdm(total=size, desc=f"Sample {sample}", unit="kept", unit_scale=True,
              dynamic_ncols=True, disable=not progress) as bar:
        for path, count in zip(paths, file_rows):
            if indices.searchsorted(offset) == indices.searchsorted(offset + count):
                offset += count
                continue
            with path.open("rb") as source:
                parquet = pq.ParquetFile(source)
                bounds = offset + np.concatenate(([0], np.cumsum([
                    parquet.metadata.row_group(group).num_rows for group in range(parquet.num_row_groups)
                ], dtype=np.int64)))
                groups = np.flatnonzero(np.diff(indices.searchsorted(bounds)) > 0)
                # Stream adjacent groups together so small row groups do not limit batch size.
                for run in np.split(groups, np.flatnonzero(np.diff(groups) > 1) + 1):
                    position = bounds[run[0]]
                    read_batch_size = min(batch_size, size - int(position)) if method == "prefix" else batch_size
                    for batch in parquet.iter_batches(row_groups=run.tolist(), columns=columns,
                                                      batch_size=read_batch_size, use_threads=False):
                        stop = position + batch.num_rows
                        positions = indices[indices.searchsorted(position):indices.searchsorted(stop)]
                        source_rows += batch.num_rows
                        bar.set_postfix(source_rows=f"{source_rows:,}", refresh=False)
                        if len(positions):
                            selected = batch.take(positions - position).to_pandas(use_threads=False)
                            selected["sample_row"] = positions.astype(np.uint64)
                            parts.append(selected)
                            bar.update(len(selected))
                        position = stop
                        if method == "prefix" and position >= size:
                            break
            offset += count
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != size:
        raise ValueError(f"Input changed while sampling {sample}: expected {size} rows, got {len(frame)}")
    for path, expected in zip(paths, sources):
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Input changed while sampling: {path}")
    report = dict(files=[str(p) for p in paths], rows=size, population_rows=total,
                  sampling_method="prefix" if method == "prefix" else "uniform_without_replacement",
                  sampling_seed=seed if method == "random" else None,
                  sampling_fraction=size / total, sampling_rescale=total / size,
                  cache_file=str(cache) if cache else None, cache_hit=False)
    if method == "prefix":
        report["sampling_rescale_assumption"] = "The first K candidate rows represent the full sample."
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Publish complete files only; an interrupted write cannot become a cache hit.
        with tempfile.TemporaryDirectory(dir=cache.parent) as temporary:
            table_path = Path(temporary) / "sample.parquet"
            receipt_path = Path(temporary) / "sample.json"
            frame.to_parquet(table_path, index=False)
            receipt_path.write_text(json.dumps(dict(identity=identity, report=report), indent=2) + "\n")
            table_path.replace(cache)
            receipt_path.replace(receipt)
    return frame, report
