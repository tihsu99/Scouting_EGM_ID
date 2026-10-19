#!/usr/bin/env python3
"""Merge completed production jobs per sample, preserving normalization receipts."""

import argparse
import json
import math
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ntuplizer import safe_token


SUM_FIELDS = ("events", "sum_genweight", "sum_genweight2", "negative_events")


def merge_sample(sample, jobs, inputdir, target, destination, skip_failed, batch_size, seen_inputs):
    target.mkdir()
    output = target / "merged.parquet"
    normalization = dict.fromkeys(SUM_FIELDS, 0)
    normalization.update(generator_bin_min=None, generator_bin_max=None)
    report = {"planned": len(jobs), "included": [], "skipped": [], "rows": 0,
              "missing_normalization": []}
    event_counts = []
    sources, dataset, provenance, schema, writer = [], None, None, None, None
    try:
        for job in jobs:
            path = inputdir / sample / Path(job["output_url"]).name
            receipt_path = Path(str(path) + ".json")
            missing = [str(p) for p in (path, receipt_path) if not p.is_file()]
            if missing:
                if not skip_failed:
                    raise FileNotFoundError("Incomplete production: " + ", ".join(missing))
                report["skipped"].append({"file": path.name, "missing": missing})
                continue
            receipt = json.loads(receipt_path.read_text())
            if (receipt["sample"] != sample or receipt["dataset"] != job["dataset"]
                    or receipt["input_files"] != job["input_files"]):
                raise ValueError(f"Receipt does not match production manifest: {path}")
            if dataset is not None and dataset != receipt["dataset"]:
                raise ValueError(f"Multiple datasets for sample {sample}")
            dataset = receipt["dataset"]
            conversion = receipt.get("conversion", {})
            signature = receipt.get("provenance", {
                "converter_sha256": conversion.get("converter_sha256"),
                "id_config": conversion.get("id_config"), "physics": receipt.get("physics"),
            })
            if provenance is not None and provenance != signature:
                raise ValueError(f"Inconsistent production settings: {path}")
            provenance = signature
            for source in receipt["input_files"]:
                if source in seen_inputs:
                    raise ValueError(f"Duplicate input file: {source}")
                seen_inputs.add(source)
                sources.append(source)
            metadata = receipt.get("normalization")
            if metadata is not None and (not isinstance(metadata, dict) or set(normalization) - metadata.keys()):
                raise ValueError(f"Incomplete normalization metadata: {receipt_path}")
            event_counts.append(receipt.get("events", (metadata or {}).get("events")))
            if metadata is None:
                report["missing_normalization"].append(path.name)
            else:
                for key in SUM_FIELDS:
                    value = metadata[key]
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"Invalid normalization {key}: {path}")
                    if key != "sum_genweight" and value < 0:
                        raise ValueError(f"Negative normalization {key}: {path}")
                    normalization[key] += value
                for key, combine in (("generator_bin_min", min), ("generator_bin_max", max)):
                    value = metadata[key]
                    if value is not None:
                        if not math.isfinite(value):
                            raise ValueError(f"Invalid normalization {key}: {path}")
                        previous = normalization[key]
                        normalization[key] = value if previous is None else combine(previous, value)
            with path.open("rb") as stream:
                parquet = pq.ParquetFile(stream)
                if parquet.metadata.num_rows != receipt["rows"]:
                    raise ValueError(f"Receipt row count mismatch: {path}")
                if schema is None:
                    schema = parquet.schema_arrow
                    writer = pq.ParquetWriter(output, schema, compression="snappy")
                elif not schema.equals(parquet.schema_arrow, check_metadata=False):
                    raise ValueError(f"Incompatible Parquet schema: {path}")
                rows = 0
                for batch in parquet.iter_batches(batch_size=batch_size):
                    writer.write_table(pa.Table.from_batches([batch], schema=schema))
                    rows += batch.num_rows
                if rows != receipt["rows"]:
                    raise ValueError(f"Incomplete Parquet read: {path}")
                report["rows"] += rows
            report["included"].append(path.name)
    finally:
        if writer is not None:
            writer.close()
    if not report["included"]:
        raise ValueError(f"No completed jobs for {sample}; cannot create this sample")
    with output.open("rb") as stream:
        merged = pq.ParquetFile(stream)
        if merged.metadata.num_rows != report["rows"]:
            raise ValueError(f"Merged row count mismatch for {sample}")
    report["normalization_complete"] = not report["missing_normalization"]
    if not report["normalization_complete"]:
        normalization = None
    merged_receipt = {
        "sample": sample, "dataset": dataset, "input_files": sources,
        "output_url": str(destination / sample / output.name), "output_format": "parquet",
        "events": sum(event_counts) if all(isinstance(n, int) and n >= 0 for n in event_counts) else None,
        "rows": report["rows"],
        "output_bytes": output.stat().st_size, "normalization": normalization,
        "provenance": provenance, "merged_files": report["included"],
        "normalization_missing_files": report["missing_normalization"],
    }
    Path(str(output) + ".json").write_text(json.dumps(merged_receipt, indent=2) + "\n")
    return merged_receipt, report


def merge(config_path, inputdir, outdir, manifests, samples=None, skip_failed=False, batch_size=65536):
    inputdir, outdir = Path(inputdir).resolve(), Path(outdir).resolve()
    if not inputdir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {inputdir}")
    if inputdir == outdir or inputdir in outdir.parents or outdir in inputdir.parents:
        raise ValueError("Input and output directories must be separate")
    if outdir.exists():
        raise FileExistsError(f"Output already exists; use a new directory: {outdir}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = yaml.safe_load(Path(config_path).read_text())
    selected = {name for name, settings in config["samples"].items()
                if settings.get("enabled", True) and (not samples or name in samples)}
    if not selected or (samples and set(samples) - selected):
        raise ValueError("No selected samples, or a selected sample is missing/disabled")
    jobs = {safe_token(name, "sample name"): [] for name in sorted(selected)}
    seen_outputs = set()
    for manifest in manifests:
        for job in json.loads(Path(manifest).read_text()):
            sample = job["sample"]
            if sample not in selected:
                continue
            filename = Path(job["output_url"]).name
            key = (sample, filename)
            if not filename.endswith(".parquet") or key in seen_outputs:
                raise ValueError(f"Non-Parquet or duplicate manifest output: {key}")
            seen_outputs.add(key)
            jobs[sample].append(job)
    absent = [sample for sample, planned in jobs.items() if not planned]
    if absent:
        raise ValueError(f"Samples missing from production manifests: {absent}")
    report = {"inputdir": str(inputdir), "outputdir": str(outdir),
              "manifests": [str(Path(p).resolve()) for p in manifests],
              "skip_failed": skip_failed, "samples": {}}
    outdir.parent.mkdir(parents=True, exist_ok=True)
    seen_inputs, merged_jobs = set(), []
    with tempfile.TemporaryDirectory(prefix="." + outdir.name + "-", dir=outdir.parent) as temporary:
        staging = Path(temporary)
        print(f'{"Sample":30} {"Planned":>8} {"Merged":>8} {"Skipped":>8} '
              f'{"Rows":>12} {"Normalization":>14}', flush=True)
        for sample, planned in jobs.items():
            receipt, summary = merge_sample(sample, planned, inputdir, staging / sample,
                                             outdir, skip_failed, batch_size, seen_inputs)
            report["samples"][sample] = summary
            merged_jobs.append({key: receipt[key] for key in
                                ("sample", "dataset", "input_files", "output_url", "output_format")})
            print(f'{sample:30} {len(planned):8} {len(summary["included"]):8} '
                  f'{len(summary["skipped"]):8} {summary["rows"]:12,} '
                  f'{"complete" if summary["normalization_complete"] else "unavailable":>14}', flush=True)
        (staging / "manifest.json").write_text(json.dumps(merged_jobs, indent=2) + "\n")
        (staging / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
        if outdir.exists():
            raise FileExistsError(f"Output appeared during merge: {outdir}")
        staging.rename(outdir)
    print(f"Merged campaign: {outdir}", flush=True)
    print(f"Coverage report: {outdir / 'merge_report.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Production YAML; select enabled samples")
    parser.add_argument("--inputdir", required=True, help="Mounted EOS campaign directory")
    parser.add_argument("--outdir", required=True, help="New merged campaign directory")
    parser.add_argument("--manifests", nargs="+", required=True, help="Production farm manifest.json files")
    parser.add_argument("--samples", nargs="+", help="Subset of enabled production sample keys")
    parser.add_argument("--skip-failed", action="store_true", help="Skip jobs missing Parquet or its receipt")
    parser.add_argument("--batch-size", type=int, default=65536, help="Rows streamed per batch")
    args = parser.parse_args()
    try:
        merge(args.config, args.inputdir, args.outdir, args.manifests,
              args.samples, args.skip_failed, args.batch_size)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
