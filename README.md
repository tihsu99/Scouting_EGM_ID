# Scouting EGM ID

Produce ID-training Parquet from DAS samples that already contain scouting
products, then run the analysis. This pipeline does not run HLT reconstruction
to create scouting products from ordinary MiniAOD.

## 1. Create and build CMSSW

Run on an **EL9 CERN host with CVMFS**, such as lxplus.
Run the following steps in order. Stop on an error before continuing.
The build block starts a clean Bash process so an already sourced LCG/Coffea
environment cannot change CMSSW's Python or library paths.
These commands create a new installation under `$HOME/Scouting_EGM_ID_pipeline`.
If you already have that CMSSW directory, either choose a new `pipeline_root`
or use the [existing-area repair instructions](ntuplizer/README.md#checks-when-something-fails).

```bash
pipeline_root="$HOME/Scouting_EGM_ID_pipeline"
mkdir -p "$pipeline_root"
env -i HOME="$HOME" USER="$(id -un)" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
    /bin/bash --noprofile --norc -s -- "$pipeline_root" <<'CMSSW'
set -eo pipefail
shopt -s expand_aliases
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd "$1"
cmsrel CMSSW_15_1_0_pre3
cd CMSSW_15_1_0_pre3/src
cmsenv

git cms-init
git remote add scouting-org https://github.com/CMS-Run3ScoutingTools/cmssw.git
git fetch scouting-org asahasra_1510pre3_MakeNanoForERegress
git checkout -b codex/scouting-id-runtime 3bcde91ff6ce2e635c6672b5dfa3774a2a427986
git cms-addpkg PhysicsTools/Scouting
git cms-addpkg PhysicsTools/NanoAOD
scram b -j 4
cmsenv

test "$(git rev-parse HEAD)" = 3bcde91ff6ce2e635c6672b5dfa3774a2a427986
python3 - <<'PY'
import os
from pathlib import Path
import sys
expected = Path(os.environ["CMSSW_BASE"]) / "src/PhysicsTools/NanoAOD/python/run3scouting_cff.py"
print("Python executable:", sys.executable, flush=True)
print("CMSSW_BASE:", os.environ["CMSSW_BASE"], flush=True)
print("Source exists:", expected.is_file(), flush=True)
import PhysicsTools.NanoAOD.run3scouting_cff as scouting
loaded = Path(scouting.__file__).resolve()
print("Loaded scouting configuration:", loaded)
assert loaded == expected.resolve(), "Python is not loading the local NanoAOD package"
assert hasattr(scouting, "scoutingElectronRegressVarTable"), "Custom NanoAOD configuration is missing"
print("CMSSW CONFIGURATION OK")
PY
edmPluginHelp -p Run3ScoutingEGammaP4RegressVarProducer
edmPluginHelp -p Run3ScoutingElectronBestTrackProducer
echo "CMSSW SETUP PASSED"
CMSSW
```

Continue only after `CMSSW SETUP PASSED`. This checks the pinned Git revision,
the Python configuration loaded from your local package, and both C++ plugins.
Use a named branch: `git cms-addpkg` fails on a detached checkout.
Run `cmsenv` again after the first build to refresh the runtime before checking
the newly built packages. If importing fails, use the
[Python import diagnostics](ntuplizer/README.md#checks-when-something-fails).
The clean process also keeps CMSSW's environment out of the next steps.

`PhysicsTools/Scouting` supplies the C++ producers; `PhysicsTools/NanoAOD`
supplies the modified Nano tables. The `cmsRun` configuration is already in
this repository at `ntuplizer/nano_cfg.py`.

## 2. Get this repository and configure the sample

In the same outer shell, clone this repository outside CMSSW:

```bash
git clone https://github.com/tihsu99/Scouting_EGM_ID.git "$pipeline_root/Scouting_EGM_ID"
cd "$pipeline_root/Scouting_EGM_ID"
```

If you already have this repository, `cd` to that checkout instead of cloning.

Edit [`config/ntuplizer.yaml`](config/ntuplizer.yaml):

- `cmssw.base`: the full path printed by `echo "$pipeline_root/CMSSW_15_1_0_pre3"`, without `/src`.
- `samples.<name>.das`: the scouting-enabled MINIAODSIM dataset to process.
- `production.eos_output`: your EOS output URL.
- `production.campaign`: use `id_parquet_smoke_v1` for the first test.

Keep `production.output_format: parquet` to run Nano production and ID conversion
in one job. `workflow.yaml` controls the stored columns and gen matching.
`conversion.setup_script` selects the Coffea environment; it defaults to `env.sh`.

The example sample key below is `DYToLL_M50`; replace it in both commands if
you choose another key. The configured Winter25 DY sample passed the scouting
modules on the first two tested events. That run then stopped because the MC
configuration lacked `PDTRecord`; `nano_cfg.py` now loads its particle data table.
Repeat the complete local test before production. The earlier Winter22 sample
lacked scouting electrons and `hltScoutingPFPacker:rho`; changing the era modifier
does not create missing input collections.

CMSSW remains in your home directory. Condor writes temporary Nano and Parquet
files in worker scratch and uploads the final Parquet to EOS. The worker must
be able to read your CMSSW directory.

## 3. Submit

### 3.1 Prepare one file and 100 events

Run from the **Scouting_EGM_ID repository**, in the outer Bash shell from
steps 1–2. If continuing in a new shell, first `cd` to this repository.

```bash
source env.sh
python3 -c 'import yaml, id_ntupler, pyarrow.parquet; print("CONVERTER IMPORTS OK")'
voms-proxy-init --rfc --voms cms -valid 192:00
export X509_USER_PROXY="$(voms-proxy-info -path)"

python3 ntuplizer.py config/ntuplizer.yaml \
    --farm-dir farms/id_smoke --samples DYToLL_M50 \
    --max-files 1 --max-events 100
```

Continue only if the imports, proxy creation, and farm preparation succeed.
This only generates the farm. Use a new farm directory for each run; replace
`farms/id_smoke` consistently below if it already exists.

### 3.2 Run the complete local test

With `output_format: parquet`, run the
prepared job interactively. Replace `DYToLL_M50` above and below if testing
another sample. CMSSW uses your compiled home area; data and caches stay in a
new `/tmp` directory. The clean shell keeps the submit-side LCG environment out
of CMSSW; the converter activates its own environment automatically.

```bash
farm="$PWD/farms/id_smoke"
local_test=$(mktemp -d /tmp/scouting-id.XXXXXX)
echo "Local test directory: $local_test"
set -o pipefail
env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 X509_USER_PROXY="$X509_USER_PROXY" \
    /bin/bash --noprofile --norc -s -- "$farm" "$local_test" <<'BASH' 2>&1 | tee "$local_test/local_test.log"
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

### 3.3 Check the result and clean up

Proceed only after `LOCAL TEST PASSED` and a zero exit status from the block.
The test checks the ROOT `Events` tree and required branches, then runs the ID
converter and checks nonempty Parquet output, required columns, and readability.

| Local file | What to inspect |
| --- | --- |
| `$local_test/local_test.log` | CMSSW and converter messages, including any failure |
| `$local_test/FrameworkJobReport.xml` | CMSSW job report |
| `$local_test/scouting_nano.root` | Intermediate NanoAOD `Events` tree |
| `$local_test/parquet/output_1.parquet` | Final electron rows for the ID workflow |
| `$local_test/conversion_report.json` | Row count, columns, and converter versions |

Nothing is uploaded to EOS. Keep the directory if the test fails. An empty
electron table may require a larger `--max-events` test. After changing YAML or
pipeline code, generate a new farm and rerun in a new temporary directory.
After repairing only the shared CMSSW installation, reuse the farm and rerun
step 3.2. See [troubleshooting](ntuplizer/README.md#checks-when-something-fails)
for missing producers or input products.

After inspecting a successful test, you can remove its temporary files:

```bash
rm -r -- "${local_test:?Local test directory is not set}"
unset local_test
```

This leaves the farm and your compiled CMSSW installation available for Condor.

### 3.4 Submit the tested job to Condor

Check worker access and EOS upload using the same prepared job:

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

Current limitations: the Winter25 sample still needs a complete Nano-to-Parquet
test after the `PDTRecord` fix; the converter also has an unresolved
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
