#!/usr/bin/env python3
"""Standalone correctionlib ID application; this file can accompany the JSON."""
import argparse
import os
import tempfile

import correctionlib
import numpy as np


def evaluate_id(corrections, name, inputs):
    """Evaluate a WP on a mapping or DataFrame of raw columns; no calibration step."""
    correction = corrections[name]
    arguments = [np.asarray(inputs[item.name], dtype=np.float64) for item in correction.inputs]
    result = np.asarray(correction.evaluate(*arguments))
    if not np.isin(result, [0., 1.]).all():
        raise ValueError(f"{name} did not return a binary ID decision")
    return result == 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correctionset", help="Exported scouting_id.json or .json.gz")
    parser.add_argument("--input", required=True, help="Candidate-level Parquet with raw input columns")
    parser.add_argument("--output", required=True, help="New Parquet retaining inputs and adding pass columns")
    parser.add_argument("--wps", nargs="+", help="Correction names; default: all exported WPs")
    parser.add_argument("--batch-size", type=int, default=200000, help="Candidates processed per batch")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    from pathlib import Path
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    corrections = correctionlib.CorrectionSet.from_file(args.correctionset)
    names = args.wps or list(corrections)
    raw_columns = sorted({item.name for name in names for item in corrections[name].inputs})
    source = pq.ParquetFile(args.input)
    for name in names:
        if f"pass_{name}" in source.schema_arrow.names:
            raise ValueError(f"Output column already exists: pass_{name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Batch the full candidate table and publish only a complete output file.
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    writer = None
    try:
        tables = (pa.Table.from_batches([batch]) for batch in source.iter_batches(batch_size=args.batch_size))
        if source.metadata.num_rows == 0:
            tables = [source.read()]
        for table in tables:
            frame = table.select(raw_columns).to_pandas()
            for name in names:
                table = table.append_column(f"pass_{name}", pa.array(evaluate_id(corrections, name, frame)))
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
        writer.close()
        writer = None
        os.replace(temporary, output)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
