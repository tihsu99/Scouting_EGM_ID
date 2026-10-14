# Scouting EGM ID

Produce ID-training Parquet from DAS samples, then run the analysis.

## 1. Set up the ntuplizer in your CMSSW area

In an EL9 shell, use your existing CMSSW directory (`cmssw.base` in the YAML).
For the first setup in a clean CMSSW source checkout:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd /afs/cern.ch/user/t/tihsu/Scouting_EGM_ID_pipeline/CMSSW_15_1_0_pre3/src
cmsenv

git cms-init
git remote add scouting-org https://github.com/CMS-Run3ScoutingTools/cmssw.git
git fetch scouting-org asahasra_1510pre3_MakeNanoForERegress
git checkout --detach 3bcde91ff6ce2e635c6672b5dfa3774a2a427986
git cms-addpkg PhysicsTools/Scouting
git cms-addpkg PhysicsTools/NanoAOD
scram b -j 4
```

Skip `git cms-init` and `git remote add` if already done. If the scouting code
is already installed, just `cmsenv` and rebuild after your changes.
The checkout selects the reference revision used by this pipeline.

`PhysicsTools/Scouting` supplies the C++ producers; `PhysicsTools/NanoAOD`
supplies the modified Nano tables. The `cmsRun` configuration is already in
this repository at `ntuplizer/nano_cfg.py`.

## 2. Edit the production YAML

Edit [`config/ntuplizer.yaml`](config/ntuplizer.yaml):

- `cmssw.base`: your compiled CMSSW directory, without `/src`.
- `samples.<name>.das`: the DAS dataset to process.
- `production.eos_output`: your EOS output URL.
- `production.campaign`: use `id_parquet_smoke_v1` for the first test.

Keep `production.output_format: parquet` to run Nano production and ID conversion
in one job. `workflow.yaml` controls the stored columns and gen matching.
`conversion.setup_script` selects the Coffea environment; it defaults to `env.sh`.

CMSSW remains in your home directory. Condor writes temporary Nano and Parquet
files in worker scratch and uploads the final Parquet to EOS. The worker must
be able to read your CMSSW directory.

## 3. Submit

From the **Scouting_EGM_ID repository**, in a fresh Bash shell on an EL9 host
with CVMFS (for example, lxplus), first prepare one small job:

```bash
source env.sh
voms-proxy-init --rfc --voms cms -valid 192:00
export X509_USER_PROXY="$(voms-proxy-info -path)"

python3 ntuplizer.py config/ntuplizer.yaml \
    --farm-dir farms/id_smoke --samples DYToLL_M50 \
    --max-files 1 --max-events 100
```

This only generates the farm. Use a new farm directory for each run.

**Test locally before submitting.** With `output_format: parquet`, run the
prepared job interactively. Replace `DYToLL_M50` above and below if testing
another sample. CMSSW uses your compiled home area; data and caches stay in a
new `/tmp` directory. The clean shell keeps the submit-side LCG environment out
of CMSSW; the converter activates its own environment automatically.

```bash
farm="$PWD/farms/id_smoke"
local_test=$(mktemp -d /tmp/scouting-id.XXXXXX)
echo "Local test directory: $local_test"
env -i HOME="$HOME" PATH=/usr/bin:/bin X509_USER_PROXY="$X509_USER_PROXY" \
    /bin/bash --noprofile --norc -s -- "$farm" "$local_test" <<'BASH'
set -eo pipefail
cp "$1"/{runtime.env,nano_cfg.py,worker.py,convert_id.py,id_ntupler.py,id_config.yaml,conversion_env.sh} "$2/"
cp "$1/jobs/DYToLL_M50_00001.json" "$2/job.json"
cd "$2"
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
source runtime.env
source "$CMSSET_DEFAULT"
cd "$CMSSW_AREA/src"
cmssw_runtime=$(scram runtime -sh)
eval "$cmssw_runtime"
test "$CMSSW_VERSION" = "$CMSSW_RELEASE"
cd "$2"
cmsRun -j FrameworkJobReport.xml nano_cfg.py job=job.json
python3 - <<'PY'
import json
from pathlib import Path
from worker import inspect_root, convert_to_parquet
job = json.loads(Path("job.json").read_text())
print("Nano events:", inspect_root("scouting_nano.root", job["required_branches"]))
output, report = convert_to_parquet(Path("job.json"), job)
print(json.dumps(report, indent=2))
print("LOCAL TEST PASSED:", output.resolve())
PY
BASH
```

Proceed only after `LOCAL TEST PASSED`. Inspect `$local_test/scouting_nano.root`,
`$local_test/parquet/output_1.parquet`, and `$local_test/conversion_report.json`.
Nothing is uploaded to EOS; files remain for debugging. If the test fails, fix
the error first. An empty electron table may require a larger `--max-events`
test. After changing YAML or pipeline code, generate a new farm to test the
updated snapshot.

**Submit the tested job to Condor** to check worker access and EOS upload:

```bash
condor_submit -dry-run farms/id_smoke/submit.ads farms/id_smoke/submit.sub && \
    condor_submit farms/id_smoke/submit.sub
condor_q
```

Logs are in `farms/id_smoke/logs/`; a successful job prints `COMPLETE`.
The local test checks processing; the Condor test also checks worker access,
resource limits, and EOS stage-out.

After the smoke job succeeds, change `production.campaign` to `id_parquet_v1`,
keep `max_events: -1` in the YAML, and submit without the smoke limits:

```bash
python3 ntuplizer.py config/ntuplizer.yaml \
    --farm-dir farms/id_parquet_v1 --submit
```

`--submit` generates the farm, checks the Condor submit file, and submits it.

The output is `<eos_output>/<campaign>/<sample>/output_N.parquet`.
After all jobs finish, copy `inputdir` from `farms/id_parquet_v1/training_input.yaml`
into `config/standard.yaml`, check the sample names, and run section 4.
There is no separate `condor.py` step in this combined mode.

Current limitations: the configured Winter22 sample has not been checked against
the required 2024 scouting fields; the converter also has an unresolved
out-of-cone truth-label overwrite. See [validation and troubleshooting](ntuplizer/README.md)
before treating the output as validated training data.

## 4. Step2: Luigi workflow

Run analysis pipeline:

```bash
python3 analysis.py --config-path config/standard.yaml --local-scheduler
```

How it is dispatched in your code (`analysis.py`):
- For each entry under `workflow:`, Luigi imports `workflow/<ModuleName>.py`.
- It calls function `<ModuleName>Analysis(**kwargs)` as the module entrypoint.
- Example: `workflow.InspectDistribution` -> `workflow/InspectDistribution.py` -> `InspectDistributionAnalysis(**kwargs)`.
- If `skip: True` is set for a module, that module is not run.
- Modules are chained sequentially in YAML order.

How arguments are built for each module:
- Global args = all top-level keys in YAML except `workflow`.
- Module args = `workflow.<ModuleName>` block.
- Final kwargs passed to module = `global args` then overwritten by `module args`.
- If `n_workers` is still missing, default is `8`.

Step2 config demo (`config/standard.yaml` style):

```yaml
inputdir: /path/to/step1_parquet
process:
  signal:
    requirement: "((status == 1) | (status == 3)) & (pt > 10)"
    samples:
      DYToLL_M50: 1.0
  background:
    requirement: "((status == 0) | (status == 2)) & (pt > 10)"
    samples:
      TTTo2L2Nu: 1.0

region_definition:
  Barrel:
    requirement: "(abs(eta) < 1.47)"
  Endcap:
    requirement: "(abs(eta) > 1.47) & (abs(eta) < 2.5)"

n_files: 4
n_workers: 8

workflow:
  DeriveEffectiveArea:
    skip: false
    plotdir: /path/to/output/EffectiveArea
    requirement: "((status == 1) | (status == 3)) & (pt > 10)"
    # Module-specific n_workers overrides global n_workers
    n_workers: 4
    EffectiveArea: {}

  InspectDistribution:
    skip: false
    plotdir: /path/to/output/plot_distribution
    storedir: /path/to/output/data
    kinematic_reweighting: BACKGROUND
    regions:
      - Barrel
      - Endcap
    inspect_variable:
      pt: [0, 200, 40]
      eta: [-3, 3, 24]

  TrainCutBasedID_TMVA:
    skip: false
    outputdir: /path/to/output/train_results
    regions:
      - Barrel
      - Endcap
    target_efficiencies: [0.95, 0.9, 0.8]
    variables: {}

  ValidateID:
    skip: false
    outputdir: /path/to/output/performance
    regions:
      - Barrel
      - Endcap
    monitor_eff:
      pt:
        binning: [0, 5, 10, 20, 40, 60, 80, 100, 200, 500]
    variables: {}
    wp_sources: {}
```

Replace all placeholder paths with your own directories (do not keep `tihsu` paths).

## 5. Per-module purpose (Step2)

- `analysis.py`: Luigi entrypoint; reads YAML, merges global/module args, and dispatches `<ModuleName>Analysis`.
- `workflow/DeriveEffectiveArea.py` (`DeriveEffectiveAreaAnalysis`): derive EA/correction terms.
- `workflow/InspectDistribution.py` (`InspectDistributionAnalysis`): produce and inspect train/valid datasets.
- `workflow/TrainCutBasedID_TMVA.py` (`TrainCutBasedID_TMVAAnalysis`): optimize cut-based WP with TMVA.
- `workflow/ValidateID.py` (`ValidateIDAnalysis`): evaluate WP efficiency/performance plots.
