#!/bin/bash
set -eo pipefail

# The launcher must never fall back to a submit-host directory for large files.
: "${_CONDOR_SCRATCH_DIR:?This worker must run inside Condor scratch}"
cd "$_CONDOR_SCRATCH_DIR"
job="$1"
source runtime.env
work=$(mktemp -d "$_CONDOR_SCRATCH_DIR/ntuplizer.XXXXXX")
trap 'rc=$?; echo "Worker exit code: $rc"; rm -rf "$work"; exit "$rc"' EXIT
export TMPDIR="$work/tmp"
export XDG_CACHE_HOME="$work/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cp "$job" "$work/job.json"
cp nano_cfg.py worker.py "$work/"
if [ -f convert_id.py ]; then
    cp convert_id.py id_ntupler.py truth_matching.py id_config.yaml conversion_env.sh "$work/"
fi
source "$CMSSET_DEFAULT"
if [ ! -d "$CMSSW_AREA/src" ]; then
    echo "CMSSW area is not accessible from this worker: $CMSSW_AREA" >&2
    exit 1
fi
cd "$CMSSW_AREA/src"
cmssw_runtime=$(scram runtime -sh)
eval "$cmssw_runtime"
test "$CMSSW_VERSION" = "$CMSSW_RELEASE"
cd "$work"
python3 worker.py job.json
