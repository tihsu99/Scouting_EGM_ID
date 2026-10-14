# Scouting EGM ID

A quick-start guide to run the full pipeline.

## 1. Clone the repository

```bash
git clone https://github.com/tihsu99/Scouting_EGM_ID.git
cd Scouting_EGM_ID
```

## 2. Environment setup

Load the CVMFS/LCG environment:

```bash
source env.sh
```

Create a CMS proxy:

```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

Important before running:
- Replace user-specific paths (i.e. containing `tihsu`) in configs.
- At minimum check `workflow.yaml` and `config/standard.yaml`.

## 3. Produce ID training ntuples from DAS

Run this stage on the CERN submission host. The combined pipeline is:

```text
DAS MINIAODSIM -> CMSSW NanoAOD in Condor scratch -> ID Parquet -> EOS
```

CMSSW stays in your maintained home/shared directory. Each worker activates that
area, returns to Condor scratch, runs both processing stages, and uploads only
the Parquet output and its JSON receipt. No CMSSW archive is required.

### 3.1 Prepare the CMSSW ntuplizer once

An ordinary CMSSW release is not enough: your area needs the scouting fork's
compiled `Run3ScoutingEGammaP4RegressVarProducer` and modified NanoAOD tables.
Follow [the CMSSW package setup commands](ntuplizer/README.md#install-the-scouting-ntuplizer-in-your-maintained-area)
to add `PhysicsTools/Scouting` and `PhysicsTools/NanoAOD` to your area and compile.
If they are already installed, use the plugin/configuration checks there instead.
The reference setup is `CMSSW_15_1_0_pre3`, `el9_amd64_gcc12`.

The entry points have different roles:

| File | Role |
| --- | --- |
| `ntuplizer.py` | Query DAS, split files, generate and optionally submit Condor jobs |
| `ntuplizer/nano_cfg.py` | CMSSW configuration passed to `cmsRun`; already supplied here |
| `id_ntupler.py` | Convert the temporary Nano into electron-level ID Parquet |
| `config/ntuplizer.yaml` | Samples, shared CMSSW path, EOS output and job resources |
| `workflow.yaml` | ID columns and gen-matching settings used during conversion |

You do not need to copy `nano_cfg.py` into CMSSW or run `run.sh` manually.
Condor transfers the configuration and launches the worker.

### 3.2 Configure and check the submission environment

Use a fresh shell in this repository, separate from the shell where you activated
CMSSW. The existing converter requires a compatible Coffea environment. Run:

```bash
source env.sh
python3 -c 'import yaml, id_ntupler; import pyarrow.parquet; print("Submission and conversion imports OK")'
voms-proxy-init --rfc --voms cms -valid 192:00
export X509_USER_PROXY="$(voms-proxy-info -path)"
command -v dasgoclient condor_submit xrdcp xrdfs
cp -n config/ntuplizer.yaml config/ntuplizer.local.yaml
```

`cp -n` preserves an existing local configuration. Edit that local file; do not
replace the whole YAML with the partial settings below.

| Field | What to set/check |
| --- | --- |
| `cmssw.base` | Your compiled release directory, without `/src`; same absolute path must work on workers |
| `cmssw.release`, `scram_arch` | Must match that area; default EL9 architecture is `el9_amd64_gcc12` |
| `cmssw.pset` | Keep `../ntuplizer/nano_cfg.py` for the supplied Nano recipe |
| `samples.<name>.das` | Full DAS name of a sample containing the required scouting products |
| `production.eos_output` | Writable `root://eosuser.cern.ch//eos/user/...` directory |
| `production.campaign` | Use `id_parquet_smoke_v1` for the smoke test |
| `production.output_format` | Keep `parquet` to combine Nano production and conversion |
| `production.id_workflow` | Keep `../workflow.yaml` or point to your ID-column configuration |
| `conversion.setup_script` | Self-contained wrapper for your validated Coffea environment; default `../env.sh` |
| `condor.cpus`, `memory_mb`, `disk_mb` | Per-job resources; converter workers must not exceed requested CPUs |

The DAS samples in `config/ntuplizer.local.yaml` select the production inputs.
The old sample paths under `workflow.yaml` are ignored by this combined stage;
only its ID columns and matching settings are used. YAML-relative paths are
resolved relative to the YAML file.

**Input check:** the current example selects a `Run3Winter22MiniAOD` dataset,
while the default modifier and requested electron fields describe 2024 scouting.
This input has not been validated. A `MINIAODSIM` dataset name does not guarantee
that it contains scouting objects or the required shower variables. Follow the
[input-product check](ntuplizer/README.md#check-the-input-products-before-submission)
before submitting. Do not change the era alone and assume all requested fields
will become available.

### 3.3 Prepare, inspect and submit one smoke job

After setting a distinct smoke campaign in the local YAML:

```bash
python3 ntuplizer.py config/ntuplizer.local.yaml \
    --farm-dir farms/id_smoke --samples DYToLL_M50 \
    --max-files 1 --max-events 100

cat farms/id_smoke/jobs/DYToLL_M50_00001.json
cat farms/id_smoke/submit.sub
condor_submit -dry-run farms/id_smoke/submit.ads farms/id_smoke/submit.sub
condor_submit farms/id_smoke/submit.sub
condor_q
```

Replace `DYToLL_M50` if you use a different sample key. Preparation without
`--submit` does not submit anything. Existing farm directories are rejected;
choose a new farm for each revised test. The sample/event limits apply per sample
and per job, respectively.

Stop if the imports, input checks, preparation or Condor dry-run fail. For held
jobs, use `condor_q -hold` and inspect `farms/id_smoke/logs/`. A successful worker
prints `COMPLETE` in its `.out` log and exits with code zero. Then inspect the
Parquet rows, columns and `status` distribution as described in the
[smoke-result checks](ntuplizer/README.md#3-check-the-smoke-result-then-run-all-events).
The expected output URL is in `farms/id_smoke/manifest.json`.

### 3.4 Full production and training handoff

Once the smoke result is understood, change `production.campaign` to a new value,
for example `id_parquet_v1`, keep `production.max_events: -1`, and omit both
smoke limits:

```bash
python3 ntuplizer.py config/ntuplizer.local.yaml \
    --farm-dir farms/id_parquet_v1 --submit
```

`--submit` runs the Condor dry-run before submitting. Keep your shared CMSSW area
unchanged while jobs are queued or running. After all expected jobs pass, EOS
contains:

```text
<eos_output>/<campaign>/<sample>/output_1.parquet
<eos_output>/<campaign>/<sample>/output_1.parquet.json
```

Copy `inputdir` from `farms/id_parquet_v1/training_input.yaml` into your full
analysis YAML, then continue to section 4. This generated file is only a path
fragment. The analysis needs a mounted EOS/local directory and matching sample
names under `process`; it does not accept an XRootD URL as `inputdir`.
Review `n_files`: the current analysis configuration limits input to four files
per sample, which is not automatically the full production.

**Validation boundary:** local orchestration tests do not establish successful
CMSSW/Coffea processing on your dataset. There is also an unresolved issue in the
existing ID converter: parent-based status assignments can overwrite an
out-of-cone unmatched label. Resolve/test the truth-label semantics before using
these files as validated training inputs. Combining the stages preserves that
existing implementation.

### 3.5 Optional: convert already-existing Nano ROOT files

Skip this subsection when using the combined `output_format: parquet` mode.
The following older stage is for Nano ROOT files that already exist:

Run Condor submission:

```bash
python3 condor.py workflow.yaml --store_path <OUTPUT_DIR> --chunks 20 --farm_dir chunks --submit
```

#### Arguments for the standalone converter

- `workflow.yaml`: input config for samples, proxy, and stored branches.
- `--store_path`: output directory for produced parquet files.
- `--chunks`: number of file chunks (roughly number of jobs per sample).
- `--farm_dir`: where chunk text files and Condor scripts are written.
- `--submit`: actually submit jobs; without this flag only submission files are prepared.

Main fields inside `workflow.yaml`:
- `samples.<name>.path`: dataset directory (without AAA prefix).
- `samples.<name>.fname`: ROOT file base name (e.g. `scouting_nano.root`).
- `samples.<name>.numbers`: total file count.
- `proxy_file`: your proxy path.
- `gen_matching_dR`: reco/gen matching cone.
- `ntupler_store_values_for_event`: event-level branches to store.
- `ntupler_store_values_for_electron`: electron-level branches to store.

Example `workflow.yaml` template:

```yaml
samples:
  DYToEE_M0p1to4:
    path: /store/user/<your_user>/<your_dataset>/0000/
    fname: scouting_nano.root
    numbers: 8
  DYToLL_M50:
    path: /store/user/<your_user>/<your_dataset>/0000/
    fname: scouting_nano.root
    numbers: 172

gen_matching_dR: 0.1

proxy_file: /afs/cern.ch/user/<first_letter>/<your_user>/tmp/x509up

ntupler_store_values_for_event:
  Pileup:
    - nPU
  ScoutingRho:
    - fixedGridRhoFastjetAll

ntupler_store_values_for_electron:
  - pt
  - eta
  - phi
  - hOverE
  - sigmaIetaIeta
  - status
```

Replace placeholder paths with your own directories and username (do not keep `tihsu` paths).

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
