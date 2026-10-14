"""Run CMSSW, verify the ID schema, and publish a checksummed output to EOS."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit
import uuid


def run(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True)


def inspect_root(path, required):
    import ROOT

    handle = ROOT.TFile.Open(str(path))
    if not handle or handle.IsZombie() or handle.TestBit(ROOT.TFile.kRecovered):
        raise RuntimeError(f"Invalid or recovered ROOT file: {path}")
    try:
        tree = handle.Get("Events")
        if not tree or not tree.InheritsFrom("TTree"):
            raise RuntimeError("Missing NanoAOD Events tree")
        missing = set(required) - {branch.GetName() for branch in tree.GetListOfBranches()}
        if missing:
            raise RuntimeError("Missing ID input branches: " + ", ".join(sorted(missing)))
        entries = int(tree.GetEntries())
        if entries == 0:
            raise RuntimeError("Output has no events; investigate before accepting this job")
        for index in {0, entries - 1}:
            if tree.GetEntry(index) <= 0:
                raise RuntimeError(f"Unreadable event {index}")
        return entries
    finally:
        handle.Close()


def stage_out(source, destination, attempts):
    parsed = urlsplit(destination)
    endpoint = f"root://{parsed.netloc}"
    remote = "/" + parsed.path.lstrip("/")
    temporary = remote + ".partial-" + uuid.uuid4().hex
    run("xrdfs", endpoint, "mkdir", "-p", str(Path(remote).parent))
    # Never overwrite a completed result, including when a job is rerun.
    existing = subprocess.run(["xrdfs", endpoint, "stat", remote], capture_output=True, text=True)
    if existing.returncode == 0:
        raise RuntimeError(f"Output already exists; use a new campaign or inspect the existing file: {destination}")
    try:
        for attempt in range(attempts):
            try:
                run("xrdcp", "--nopbar", "--force", "--cksum", "adler32", source, endpoint + "/" + temporary)
                break
            except subprocess.CalledProcessError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(5 * (attempt + 1))
        run("xrdfs", endpoint, "mv", temporary, remote)
    except BaseException:
        subprocess.run(["xrdfs", endpoint, "rm", temporary], check=False)
        raise


def convert_to_parquet(job_path, job):
    work = Path.cwd()
    home = work / "conversion_home"
    home.mkdir(exist_ok=True)
    # Source Coffea in a fresh environment, without CMSSW's Python or libraries.
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(home),
        "TMPDIR": str(work / "tmp"), "XDG_CACHE_HOME": str(work / "cache"),
        "MPLCONFIGDIR": str(work / "cache/matplotlib"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    }
    for name in ("X509_USER_PROXY", "X509_CERT_DIR"):
        if name in os.environ:
            env[name] = os.environ[name]
    subprocess.run([
        "/bin/bash", "--noprofile", "--norc", "-c",
        'set -eo pipefail; source ./conversion_env.sh; exec "$1" convert_id.py "$2"',
        "convert-id", job["conversion"]["python"], str(job_path),
    ], env=env, check=True)
    metadata = json.loads(Path("conversion_report.json").read_text())
    return Path("parquet") / f"output_{job['index']}.parquet", metadata


def main():
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text())
    run("cmsRun", "-j", "FrameworkJobReport.xml", "nano_cfg.py", f"job={job_path}")
    output = Path("scouting_nano.root")
    entries = inspect_root(output, job["required_branches"])
    metadata = {}
    if job.get("output_format", "nano") == "parquet":
        output, metadata = convert_to_parquet(job_path, job)
        # The intermediate Nano never leaves worker scratch in combined mode.
        Path("scouting_nano.root").unlink()
    report = dict(job, events=entries, output_bytes=output.stat().st_size, **metadata)
    Path("receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    stage_out(output, job["output_url"], job["stageout_attempts"])
    stage_out("receipt.json", job["output_url"] + ".json", job["stageout_attempts"])
    print("COMPLETE " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
