"""Run the existing ID converter and validate its Parquet output in Coffea's Python."""

from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys


def normalization_metadata(path):
    """Count all processed events, including events with no selected candidates."""
    import numpy as np
    import uproot

    result = {"events": 0, "sum_genweight": 0.0, "sum_genweight2": 0.0,
              "negative_events": 0, "generator_bin_min": None, "generator_bin_max": None}
    with uproot.open(path) as source:
        for chunk in source["Events"].iterate(["Generator_weight", "Generator_binvar"], library="np"):
            weights = chunk["Generator_weight"].astype(float)
            bins = chunk["Generator_binvar"]
            if not np.isfinite(weights).all() or not np.isfinite(bins).all():
                raise ValueError("Non-finite generator metadata")
            result["events"] += len(weights)
            result["sum_genweight"] += float(weights.sum())
            result["sum_genweight2"] += float(np.square(weights).sum())
            result["negative_events"] += int((weights < 0).sum())
            if len(bins):
                lo, hi = float(bins.min()), float(bins.max())
                result["generator_bin_min"] = lo if result["generator_bin_min"] is None else min(lo, result["generator_bin_min"])
                result["generator_bin_max"] = hi if result["generator_bin_max"] is None else max(hi, result["generator_bin_max"])
    return result


def inspect_parquet(path, required_columns):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    missing = set(required_columns) - set(parquet.schema_arrow.names)
    if missing:
        raise RuntimeError("Missing ID training columns: " + ", ".join(sorted(missing)))
    rows = parquet.metadata.num_rows
    # Empty candidate tables still contribute to the event normalization.
    for index in {0, parquet.num_row_groups - 1}:
        parquet.read_row_group(index, columns=required_columns)
    return {"rows": rows, "columns": parquet.schema_arrow.names}


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    settings = job["conversion"]
    subprocess.run([
        sys.executable, "id_ntupler.py", "id_config.yaml",
        "--files", str(Path("scouting_nano.root").resolve()),
        "--store_path", "parquet", "--index", str(job["index"]),
        "--cpu", str(settings["workers"]), "--chunk-size", str(settings["chunk_size"]),
        "--strict-inputs",
    ], check=True)
    report = inspect_parquet(Path("parquet") / f"output_{job['index']}.parquet", settings["required_columns"])
    report["normalization"] = normalization_metadata("scouting_nano.root")
    report["conversion_versions"] = {name: version(name) for name in
                                     ("coffea", "awkward", "uproot", "pandas", "pyarrow")}
    Path("conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
