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

The YAML also enables Winter25 `TT`, `TT_ext1`, `ZprimeTo2E_M6000`, and ten
unfiltered QCD pT samples. See [sample selection](ntuplizer/README.md#winter25-sample-selection)
for the bin coverage, extension handling, and weighting limits. Omitting
`--samples` processes all enabled samples; the smoke commands below select only DY.

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
converter and checks the Parquet schema, required columns, and readability.
Check that the smoke run contains electron rows before assessing ID distributions.
Empty electron tables are retained in production to preserve event normalization.

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
test after the `PDTRecord` and truth-label fixes. The current selections use prompt
and direct tau-decay electrons as signal, and unmatched or other nonprompt
electrons as background, within the configured sample lists. Regenerate old
Parquet outputs with the corrected converter before training; see
[truth labels and validation](ntuplizer/README.md#electron-truth-labels).

## 4. Post-ntuple ID studies

Use `postprocess.py` for the complete campaign: DY versus TT, DY versus stitched
QCD, and independent Zprime validation. It calls the existing correction,
preparation, TMVA, and validation modules. Customize sample lists, cross sections,
kinematic binning, pT regions, and stage options in
[`config/postprocess.yaml`](config/postprocess.yaml) and its base
[`config/standard.yaml`](config/standard.yaml).

Plots use [mplhep's CMS style](https://mplhep.readthedocs.io/en/latest/guide_styling/)
for ROOT-like fonts, inward ticks on all four sides, and CMS labels. Install it
in the analysis Python environment with `python3 -m pip install mplhep` if absent.
The shared `plot_style` block in `config/standard.yaml` controls Matplotlib
`rc_params`, `legend` options, and the arguments to `hep.cms.label`. The supplied MC campaign uses
**CMS Preliminary** and **13.6 TeV**; the status text is used literally without
automatically adding "Simulation". Change `cms_label.com` for a
different collision energy, or set it to `null` to omit the energy. These settings
apply to correction fits, distributions, QCD stitching, both trainers, ROC curves,
efficiency/fake-rate plots, and N-minus-one plots, including worker processes.
Use `shared.plot_style` in `config/postprocess.yaml` to replace the style block
for a study run. Styling does not change selections, weights, or metric tables.
ROC and efficiency/fake-rate measurements use markers without connecting lines;
efficiency plots retain their error bars. Legends are placed in the lower-right corner inside the axes,
following the efficiency-plot layout in
[CMS-DP-2026/048](https://twiki.cern.ch/twiki/bin/view/CMSPublic/DP2026048#Data_scouting_Trigger_Efficiency).
`plot_style.max_series_per_panel` limits the number of efficiency/fake-rate
series per panel (default: 8). Larger scans use multiple panels in the same
output figure, retaining every working point and the full metric tables.

**Produce new ntuples before running this workflow.** The production now saves
`Generator.weight`, `Generator.binvar`, and event identifiers. Each Parquet receipt
also records the generator-weight sum over every processed event, including events
without electron candidates. Old Parquet alone cannot recover this information.
Keep the `.parquet.json` receipts beside the Parquet files when copying from EOS.

Set every selected group's `normalization.groups.<group>.cross_section_pb` to the
verified effective cross section for that exact dataset (after matching/filtering).
The template leaves these values null and fails rather than guessing them. Obtain
values from campaign metadata or run CMS
[GenXSecAnalyzer](https://opendata.web.cern.ch/docs/cms-guide-xsec) on the corresponding
MiniAOD. Do not apply a filter efficiency again to an already filtered cross section.
The `TT` group shares one cross section and the combined `TT` + `TT_ext1` denominator.

Run from this checkout in your validated Coffea/PyROOT environment. Replace the
input/output paths with your mounted EOS paths; use a new output directory after
changing the configuration. The manifest must describe the same production.

```bash
source env.sh
export MPLBACKEND=Agg
python3 postprocess.py config/postprocess.yaml \
    --inputdir /eos/user/t/tihsu/dataset/scouting_id/id_parquet_v1 \
    --outputdir /eos/user/t/tihsu/database/EGM_Scouting_ID_analysis/energy_v1 \
    --manifest farms/id_parquet_v1/manifest.json --check-only
```

Continue only after a successful check. Run the same command without `--check-only`
to execute all stages. `--studies dy_tt` or `--studies dy_qcd` selects one training
study. `--steps calibrate prepare train validate highpt` is the default; select a
subset to resume, using the same configuration and output directory. `highpt`
requires prepared held-out background tables and trained WPs from the earlier steps.
The resolved YAML is saved, and a changed configuration cannot reuse that directory.

The workflow implements:

- **QCD normalization:** event weight = effective cross section times luminosity
  times generator weight divided by the processed, preselection generator-weight
  sum. Each generator bin contributes according to its cross section; electron pT
  spectra are never rescaled to force adjacent points to agree. It checks receipt
  provenance, duplicate inputs, generator bin ranges, and bin gaps/overlaps.
  `background_stitching.pdf/json` shows per-sample and total electron pT spectra.
  The supplied QCD set starts at generator pT = 30 GeV: completeness for 10-20 GeV
  electron candidates still requires a physics check or lower generator bins.
- **Corrections:** derive rho effective areas and residual energy scaling in eta
  bins from the DY training partition. Relative isolation uses
  `E = pt*cosh(eta)`, `I/E`, `(I-rho*EA)/E`, and `(I-rho*EA-scale)/E`;
  the scale fit uses `1/E`. Raw isolation numerators are unchanged and no zero
  clipping is introduced. Fit plots and coefficient/error summaries are retained.
- **Training:** fit the configurable signal-to-background `(pt, eta)` reweighting
  on training events only. `reweighting.json` stores before/after histograms and
  unsupported phase space. Unsupported cells receive zero training weight, never
  an extrapolated edge-bin factor. `training_weight_mode: absolute` explicitly
  uses absolute physical weights for calibration/training with signed MC;
  signed physical weights remain in validation. This is a training convention,
  not a replacement for signed cross-section normalization.
- **Validation:** keep all candidates from one event in the same partition. Both
  IDs are evaluated at 10-20, 20-200, and 200-1000 GeV, separately in Barrel and
  Endcap. ROC plots show the configured cut-based WP scan as points; increase
  `target_efficiencies` for a denser scan. JSON records requested and selected TMVA
  efficiency bins; an unpopulated bin outside `efficiency_tolerance` stops training.
- **High-pT validation:** Zprime supplies independent signal candidates; each
  study reuses only its held-out TT/QCD background. Separate efficiency and
  background misidentification plots cover 200-1000 GeV. The denominator is
  reconstructed signal/background electron candidates passing the configured
  baseline, not all generated electrons or all jets.
- **Default WPs:** the checked-in `ref_wp/default_{region}.json` records
  `ScoutingElectronRecommendv1` and `ScoutingElectronCustomizev1`, traced to the
  existing `EgammaAnalysis/TnPTreeProducer/plugins/ScoutingElectronRefSelector.cc`
  with its source hash. Raw E-normalized isolation and strict thresholds are
  reevaluated alongside newly optimized IDs. The EB/EE boundary is 1.479.

`config/electron_pipeline.yaml` and `config/validate_isolated.yaml` also enable
the validation-only `OfflineMapped` reference. Its CMSSW RunIIIWinter22 V1 cuts,
effective areas, source revision, input mapping and limitations are recorded in
`ref_wp/offline_electron_winter22.yaml`. The supplied DPNote mapping is
ECAL to photon isolation, HCAL to hadron isolation, and track to charged isolation:
`relIso = (trackIso + max(0, ecalIso + hcalIso - rho*EA))/pt`.
The cuts include `relIso < C0 + Cpt/pt`, `H/E < C0 + CE/E + Cr*rho/E`,
shower shape, track matching, inverse-energy/momentum and inclusive missing hits.
Conversion veto is omitted; the YAML records the remaining reconstruction approximations.

Run `validate_id.py` with the same completed calibration, training and prepared
directories and `--steps plot --outputdir <new-directory>` to add the reference
without redoing preparation or training. Existing `--id-correctionlib` exports
remain usable; mapped references are evaluated separately on the same selected
candidates and weights. Each region writes `reference_ids.json` with formulas and
source hashes. To disable these points, set `validation.reference_sources: {}`
(or `evaluation.validation.reference_sources: {}` in the standalone pipeline).
Hollow circles reuse the online colors for Tight/WP70, Medium/WP80, Loose/WP90
and Veto/WP95. This pairing is visual; compare measured signal efficiency and
background efficiency in the ROC CSV, including the pT slices.

Output layout under the selected output directory:

| Path | Content |
| --- | --- |
| `EffectiveArea/` | Fit plots, fit summaries, correctionlib JSON |
| `dy_tt/`, `dy_qcd/` | Independent study outputs |
| `<study>/data/` | Physical weights, training weights, event-separated tables, normalization audit |
| `<study>/distributions/` | Reweighting audit and stitched background spectra |
| `<study>/train_results/<region>/` | Optimized WP JSON, ROOT files, XML weights |
| `<study>/performance/<region>/` | Inclusive and pT-sliced ROC PNG/CSV, efficiency/fake-rate PNG/CSV |
| `<study>/highpt/` | Zprime validation with the same trained/default IDs |

CSV tables include candidate counts, sum of weights, squared weights, and effective
entries where applicable. Ratio errors include the numerator/denominator covariance
for weighted subsets; zero or negative total weight gives an undefined result.
These are per-candidate statistical errors, not a treatment of all correlations
between candidates in the same event. Sparse bins and signed-weight fluctuations
must be inspected before using the result as a physics measurement.

A single-study Luigi entry point is retained for custom configurations:

```bash
python3 analysis.py --config-path config/standard.yaml --local-scheduler
```

Its standalone config must supply normalization groups if physical sample mixing
is needed; the complete study driver supplies them from `postprocess.yaml`.

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
    requirement: "((status == 1) | (status == 2)) & (pt > 10)"
    samples:
      DYToLL_M50: 1.0
  background:
    requirement: "((status == 0) | (status == 3)) & (pt > 10)"
    samples:
      TT: 1.0
      TT_ext1: 1.0

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
    requirement: "((status == 1) | (status == 2)) & (pt > 10)"
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

## 6. Local checks

Run in a Python environment with the analysis dependencies:

```bash
MPLBACKEND=Agg python3 -m unittest discover -s tests -v
MPLBACKEND=Agg python3 tests/smoke_postprocess.py /tmp/scouting-id-synthetic-check
```

The second command requires ROOT/TMVA and a new output directory. It runs synthetic
calibration, both training studies, pT-sliced validation, and Zprime validation.
Synthetic cross sections and outputs are software checks, not production WPs.
