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

## 3. Step1: Produce ntuple (Could skip if you already have ntuples)

Run Condor submission:

```bash
python3 condor.py workflow.yaml --store_path <OUTPUT_DIR> --chunks 20 --farm_dir chunks --submit
```

### 3.1 Argument explanation

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
