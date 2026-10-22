"""Bounded, uniform candidate sampling with reusable raw-column caches."""
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def load_random_sample(paths, columns, limit, seed, sample, cache_dir, batch_size, progress):
    """Sample before cuts; return original row positions and inverse inclusion probability."""
    if limit is None:
        raise ValueError("Random sampling requires a positive max_rows_per_sample")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sampling.seed must be a nonnegative integer")
    sources = []
    for path in paths:
        stat = path.stat()
        sources.append(dict(path=str(path.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    identity = dict(version=1, sample=sample, sources=sources,
                    columns=None if columns is None else list(columns), limit=limit, seed=seed)
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
    # range sampling avoids allocating a permutation of the full production.
    sample_seed = hashlib.sha256(f"{seed}:{sample}".encode()).hexdigest()
    indices = np.array(sorted(random.Random(sample_seed).sample(range(total), size)), dtype=np.int64)
    parts, offset = [], 0
    with tqdm(total=size, desc=f"Sample {sample}", unit="row", unit_scale=True,
              dynamic_ncols=True, disable=not progress) as bar:
        for path, count in zip(paths, file_rows):
            if indices.searchsorted(offset) == indices.searchsorted(offset + count):
                offset += count
                continue
            with path.open("rb") as source:
                parquet = pq.ParquetFile(source)
                for group in range(parquet.num_row_groups):
                    group_rows = parquet.metadata.row_group(group).num_rows
                    if indices.searchsorted(offset) == indices.searchsorted(offset + group_rows):
                        offset += group_rows
                        continue
                    for batch in parquet.iter_batches(row_groups=[group], columns=columns,
                                                      batch_size=batch_size, use_threads=False):
                        stop = offset + batch.num_rows
                        positions = indices[indices.searchsorted(offset):indices.searchsorted(stop)]
                        if len(positions):
                            selected = batch.take(positions - offset).to_pandas(use_threads=False)
                            selected["sample_row"] = positions.astype(np.uint64)
                            parts.append(selected)
                            bar.update(len(selected))
                        offset = stop
    frame = pd.concat(parts, ignore_index=True)
    if len(frame) != size:
        raise ValueError(f"Input changed while sampling {sample}: expected {size} rows, got {len(frame)}")
    for path, expected in zip(paths, sources):
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"Input changed while sampling: {path}")
    report = dict(files=[str(p) for p in paths], rows=size, population_rows=total,
                  sampling_method="uniform_without_replacement", sampling_seed=seed,
                  sampling_fraction=size / total, sampling_rescale=total / size,
                  cache_file=str(cache) if cache else None, cache_hit=False)
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
