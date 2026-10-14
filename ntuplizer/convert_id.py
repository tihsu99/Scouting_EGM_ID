"""Run the existing ID converter and validate its Parquet output in Coffea's Python."""

from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys


def inspect_parquet(path, required_columns):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    missing = set(required_columns) - set(parquet.schema_arrow.names)
    if missing:
        raise RuntimeError("Missing ID training columns: " + ", ".join(sorted(missing)))
    rows = parquet.metadata.num_rows
    if rows == 0:
        raise RuntimeError("No electron rows in the ID training output")
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
    report["conversion_versions"] = {name: version(name) for name in
                                     ("coffea", "awkward", "uproot", "pandas", "pyarrow")}
    Path("conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
