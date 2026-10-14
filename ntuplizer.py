#!/usr/bin/env python3
"""Prepare DAS -> scouting NanoAOD or ID Parquet jobs; submit only with --submit."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from urllib.parse import urlsplit

import yaml


HERE = Path(__file__).resolve().parent


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def safe_token(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", ".."):
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def local_path(value, base):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def submit_path(path):
    # Condor path fields are not shell strings; reject syntax needing escaping.
    value = str(path)
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise ValueError(f"Use a Condor path without spaces or special characters: {value}")
    return value


def root_url(value):
    parsed = urlsplit(value)
    if (parsed.scheme != "root" or not parsed.netloc or parsed.query or parsed.fragment
            or not parsed.path.startswith("//") or ".." in parsed.path.split("/")
            or any(c.isspace() for c in value) or "$" in value or "<" in value
            or "CHANGE_ME" in value or "REPLACE_" in value):
        raise ValueError(f"Expected an explicit root://host//absolute/path URL: {value}")
    return value.rstrip("/")


def required_branches(workflow):
    # status is created by id_ntupler.py, not stored by CMSSW.
    branches = {"run", "luminosityBlock", "event", "nScoutingElectron", "nGenPart"}
    branches.update("ScoutingElectron_" + name for name in
                    workflow["ntupler_store_values_for_electron"] if name != "status")
    branches.update("ScoutingElectron_" + name for name in ("pt", "eta", "phi", "m"))
    branches.update("GenPart_" + name for name in
                    ("pt", "eta", "phi", "mass", "pdgId", "status", "statusFlags", "genPartIdxMother"))
    for collection, names in workflow["ntupler_store_values_for_event"].items():
        branches.update(collection + "_" + name for name in names)
    return sorted(branches)


def discover_files(sample, base):
    dataset = sample["das"]
    if not re.fullmatch(r"/[^/\s]+/[^/\s]+/[^/\s]+", dataset) or "REPLACE_" in dataset:
        raise ValueError(f"Expected a full DAS dataset name: {dataset}")
    if sample.get("file_list"):
        raw = local_path(sample["file_list"], base).read_text().splitlines()
    else:
        instance = sample.get("das_instance", "prod/global")
        if not re.fullmatch(r"prod/[a-z]+", instance):
            raise ValueError(f"Invalid DAS instance: {instance}")
        result = subprocess.run(
            ["dasgoclient", "--query", f"file dataset={dataset} instance={instance}", "--limit", "0"],
            check=True, text=True, capture_output=True,
        )
        raw = result.stdout.splitlines()
    files = sorted({line.strip() for line in raw if line.strip() and not line.lstrip().startswith("#")})
    if not files or any(not re.fullmatch(r"/store/[^\s]+\.root", name) for name in files):
        raise ValueError(f"DAS/file_list must contain nonempty /store/...root LFNs for {dataset}")
    return files


def prepare(config_path, farm, selected=None, max_files=None, max_events=None):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = yaml.safe_load(config_path.read_text())
    cmssw, condor, production = (config[key] for key in ("cmssw", "condor", "production"))
    release = safe_token(cmssw["release"], "CMSSW release")
    arch = safe_token(cmssw["scram_arch"], "SCRAM_ARCH")
    cmssw_area = local_path(cmssw["base"], base)
    if not (cmssw_area / "src").is_dir():
        raise ValueError(f"Missing CMSSW source directory: {cmssw_area / 'src'}")
    cmssw["base"] = str(cmssw_area)
    eos = root_url(production["eos_output"])
    campaign = safe_token(production["campaign"], "campaign")
    redirector = root_url(production["input_redirector"] + "/placeholder")
    redirector = redirector.removesuffix("/placeholder")
    cpus = positive(condor["cpus"], "condor.cpus")
    for key in ("memory_mb", "disk_mb"):
        positive(condor[key], f"condor.{key}")
    positive(production["files_per_job"], "production.files_per_job")
    events = production.get("max_events", -1) if max_events is None else max_events
    if type(events) is not int or events == 0 or events < -1:
        raise ValueError("max_events must be -1 or a positive integer")
    if max_files is not None:
        positive(max_files, "max_files")
    proxy = submit_path(local_path(condor["proxy"], base))
    workflow_path = local_path(production["id_workflow"], base)
    workflow = yaml.safe_load(workflow_path.read_text())
    workflow["samples"] = {}
    workflow["proxy_file"] = proxy
    branches = required_branches(workflow)
    output_format = production.get("output_format", "nano")
    if output_format not in ("nano", "parquet"):
        raise ValueError("production.output_format must be nano or parquet")
    conversion = None
    if output_format == "parquet":
        settings = config["conversion"]
        conversion_setup = local_path(settings["setup_script"], base)
        if not conversion_setup.is_file():
            raise ValueError(f"Missing conversion setup script: {conversion_setup}")
        workers = positive(settings.get("workers", cpus), "conversion.workers")
        if workers > cpus:
            raise ValueError("conversion.workers must not exceed condor.cpus")
        conversion = {
            "python": settings.get("python", "python3"), "workers": workers,
            "chunk_size": positive(settings.get("chunk_size", 15000), "conversion.chunk_size"),
            "required_columns": list(workflow["ntupler_store_values_for_electron"]) + [
                collection + "." + field
                for collection, fields in workflow["ntupler_store_values_for_event"].items()
                for field in fields
            ],
            "id_config": {key: value for key, value in workflow.items() if key not in ("samples", "proxy_file")},
            "converter_sha256": hashlib.sha256((HERE / "id_ntupler.py").read_bytes()).hexdigest(),
            "setup_sha256": hashlib.sha256(conversion_setup.read_bytes()).hexdigest(),
        }
    pset = local_path(cmssw["pset"], base)
    if not pset.is_file():
        raise ValueError(f"Missing CMSSW configuration: {pset}")
    samples = {name: sample for name, sample in config["samples"].items()
               if sample.get("enabled", True) and (not selected or name in selected)}
    if not samples or (selected and set(selected) - samples.keys()):
        raise ValueError("No selected samples, or a selected sample is missing/disabled")
    positive(config["physics"]["report_every"], "physics.report_every")
    for name, values in config["physics"]["best_track"].items():
        if (not isinstance(values, list) or len(values) != 2
                or any(type(value) not in (int, float) or value < 0 for value in values)):
            raise ValueError(f"physics.best_track.{name} must contain two nonnegative numbers (EB, EE)")
    jobs = []
    for name, sample in samples.items():
        safe_token(name, "sample name")
        if sample.get("is_mc") is not True:
            raise ValueError(f"{name}: the current ID consumer requires MC GenPart and Pileup; is_mc must be true")
        if sample["das"].split("/")[-1] != "MINIAODSIM":
            raise ValueError(f"{name}: this recipe requires scouting-enabled MINIAODSIM inputs")
        files = discover_files(sample, base)
        if max_files is not None:
            files = files[:max_files]
        count = positive(sample.get("files_per_job", production["files_per_job"]), "files_per_job")
        urls = []
        for start in range(0, len(files), count):
            index = start // count + 1
            filename = f"output_{index}.parquet" if output_format == "parquet" else f"scouting_nano_{index}.root"
            url = f"{eos}/{campaign}/{name}/{filename}"
            job = {
                "sample": name, "dataset": sample["das"], "index": index,
                "input_files": [redirector.rstrip("/") + "/" + f for f in files[start:start + count]],
                "output_url": url, "max_events": events, "threads": cpus,
                "is_mc": True, "physics": copy.deepcopy(config["physics"]),
                "required_branches": branches, "cmssw": cmssw,
                "stageout_attempts": positive(production.get("stageout_attempts", 3), "stageout_attempts"),
                "output_format": output_format,
            }
            if conversion:
                job["conversion"] = conversion
            jobs.append(job)
            urls.append(url)
        workflow["samples"][name] = {"files": urls}
    farm = Path(farm).resolve()
    submit_path(farm)
    farm.mkdir(parents=True, exist_ok=False)
    (farm / "logs").mkdir()
    (farm / "jobs").mkdir()
    for filename in ("run.sh", "worker.py"):
        shutil.copy2(HERE / "ntuplizer" / filename, farm / filename)
    (farm / "run.sh").chmod(0o755)
    shutil.copy2(pset, farm / "nano_cfg.py")
    transfer_files = ["runtime.env", "nano_cfg.py", "worker.py", "jobs/$(job)"]
    if conversion:
        shutil.copy2(conversion_setup, farm / "conversion_env.sh")
        shutil.copy2(HERE / "id_ntupler.py", farm / "id_ntupler.py")
        shutil.copy2(HERE / "ntuplizer/convert_id.py", farm / "convert_id.py")
        id_config = dict(conversion["id_config"], samples={})
        (farm / "id_config.yaml").write_text(yaml.safe_dump(id_config, sort_keys=False))
        transfer_files.extend(["conversion_env.sh", "id_ntupler.py", "convert_id.py", "id_config.yaml"])
    env = {"CMSSW_RELEASE": release, "SCRAM_ARCH": arch, "CMSSW_AREA": str(cmssw_area),
           "CMSSET_DEFAULT": cmssw["cmsset_default"]}
    (farm / "runtime.env").write_text("".join(f"export {key}={shlex.quote(str(value))}\n" for key, value in env.items()))
    lines = [
        "universe = vanilla", f"initialdir = {farm}", f"executable = {farm}/run.sh",
        "arguments = $(job)", "should_transfer_files = YES", "when_to_transfer_output = ON_EXIT",
        'transfer_output_files = ""',
        "transfer_input_files = " + ",".join(str(farm / name) for name in transfer_files),
        "use_x509userproxy = True", f"x509userproxy = {proxy}", "getenv = False",
        f"request_cpus = {cpus}", f"request_memory = {condor['memory_mb']}MB",
        f"request_disk = {condor['disk_mb']}MB",
        f'+JobFlavour = "{safe_token(condor["job_flavour"], "job_flavour")}"',
        f'+WantOS = "{safe_token(condor["os"], "condor.os")}"',
        "on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)",
        "output = logs/$(job).$(ClusterId).$(ProcId).out",
        "error = logs/$(job).$(ClusterId).$(ProcId).err",
        "log = logs/cluster.$(ClusterId).log",
        "queue job from (",
    ]
    for job in jobs:
        filename = f"{job['sample']}_{job['index']:05d}.json"
        (farm / "jobs" / filename).write_text(json.dumps(job, indent=2) + "\n")
        lines.append(filename)
    lines.append(")")
    (farm / "submit.sub").write_text("\n".join(lines) + "\n")
    (farm / "resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if output_format == "nano":
        (farm / "id_workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False))
    else:
        # The analysis consumer uses os.listdir, so it needs the mounted EOS path.
        training_input = {"inputdir": "/" + urlsplit(f"{eos}/{campaign}").path.lstrip("/")}
        (farm / "training_input.yaml").write_text(yaml.safe_dump(training_input))
    (farm / "manifest.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(f"Prepared {len(jobs)} jobs in {farm}")
    print(f"EOS destination: {eos}/{campaign}")
    return farm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--farm-dir", required=True, help="New directory; existing farms are never overwritten")
    parser.add_argument("--samples", nargs="+")
    parser.add_argument("--max-files", type=int, help="Limit files per sample for a smoke run")
    parser.add_argument("--max-events", type=int, help="Override events per job for a smoke run")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    try:
        farm = prepare(args.config, args.farm_dir, args.samples, args.max_files, args.max_events)
        if args.submit:
            subprocess.run(["condor_submit", "-dry-run", str(farm / "submit.ads"), str(farm / "submit.sub")], check=True)
            subprocess.run(["condor_submit", str(farm / "submit.sub")], check=True)
        else:
            print(f"Review, then run: condor_submit -dry-run {farm}/submit.ads {farm}/submit.sub")
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
