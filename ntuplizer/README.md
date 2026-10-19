# Ntuplizer reference

For setup and submission commands, use the [main README](../README.md).

## What runs in a job

1. Activate the existing compiled area from `cmssw.base`.
2. Run `nano_cfg.py` with `cmsRun` in Condor scratch, streaming DAS inputs through XRootD.
3. Run the existing `id_ntupler.py` in the Coffea environment selected by `conversion.setup_script`.
4. Check the Parquet columns and readability, delete the temporary Nano, and upload Parquet plus its JSON receipt to EOS.

The worker does not copy or rebuild CMSSW. Keep the shared area unchanged while
jobs run. Scratch must fit the intermediate Nano and final Parquet together.
The converter accumulates a job's rows in memory, so start with one input file
per job and measure memory usage.

## Configuration

| YAML field | Meaning |
| --- | --- |
| `cmssw.base` | Existing compiled CMSSW area, accessible at the same path on workers |
| `cmssw.pset` | Configuration for `cmsRun`; default `../ntuplizer/nano_cfg.py` |
| `samples.<name>.das` | Full scouting-enabled MINIAODSIM dataset name |
| `samples.<name>.enabled` | Set `false` to skip a sample |
| `samples.<name>.file_list` | Optional file containing saved `/store/...root` LFNs instead of querying DAS |
| `production.files_per_job` | Files assigned to each job; can also be overridden per sample |
| `production.max_events` | Event limit per job; `-1` processes all events |
| `production.output_format` | `parquet` combines both stages; `nano` saves the ROOT output instead |
| `production.id_workflow` | ID columns and gen-matching settings; its old sample paths are ignored |
| `conversion.setup_script` | Shell wrapper sourcing a compatible Coffea environment |
| `conversion.workers`, `chunk_size` | Converter workers and events per chunk |
| `condor.cpus`, `memory_mb`, `disk_mb` | Per-job resources |

Paths in YAML are relative to the YAML file. The conversion wrapper is copied
to workers, so use absolute paths inside it. Conversion runs in a fresh
environment without inherited CMSSW Python/library paths.

## Winter25 sample selection

`config/ntuplizer.yaml` contains the user-supplied Winter25 datasets, enabled for
production after local validation:

| Sample keys | Purpose |
| --- | --- |
| `DYToLL_M50` | Existing prompt-electron baseline |
| `TT`, `TT_ext1` | Inclusive ttbar and its additional production statistics |
| `ZprimeTo2E_M6000` | 6 TeV Z-prime electron sample for high-pT studies |
| `QCD_Pt_30to50` through `QCD_Pt_800to1000`, `QCD_Pt_1000` | Unfiltered PYTHIA QCD background coverage |

The selected QCD bins are 30–50, 50–80, 80–120, 120–170, 170–300,
300–470, 470–600, 600–800, 800–1000 GeV, followed by the `Pt-1000` tail.
The exact DAS spelling and versions are preserved, including `Pt` capitalization
and the `30to50` sample's `v3` suffix. The generator-scale bins are adjacent;
they do not impose these bounds on reconstructed electron pT or guarantee flat
electron statistics. There is no unfiltered bin below 30 GeV in the supplied
list. Check low-pT electron coverage and high-pT statistics in the produced data.

Do not combine this set with `Pt-600`, the separate bins starting at 1400 GeV,
HT-binned MadGraph, EM/muon-enriched, or flat-pT samples without a separate
stitching prescription. The `Pt-1000` sample is selected as the open tail;
confirm its generator cuts from production provenance before normalization.

The supplied dataset names are not a completed scouting compatibility test.
Prepare and test one file per sample before full submission, for example:

```bash
python3 ntuplizer.py config/ntuplizer.yaml \
    --farm-dir farms/qcd_1000_smoke --samples QCD_Pt_1000 \
    --max-files 1 --max-events 100
```

In the main README's local-test block use `farm="$PWD/farms/qcd_1000_smoke"`
and `jobs/QCD_Pt_1000_00001.json`. Without `--samples`, preparation includes
all enabled samples. Use `enabled: false` to exclude any sample.

`TT` is inclusive ttbar, not the legacy `TTTo2L2Nu` dataset. The old
`config/standard.yaml` sample list is not automatically updated: after production,
select the new sample keys and truth categories there. Keep `TT` and `TT_ext1`
as separate output folders but normalize their combined statistics as one process,
not two copies of the ttbar cross section. Prompt ttbar electrons can be signal;
the dataset name alone is not an electron truth label.

For physical spectra, QCD bins require cross sections and generator-weight sums;
concatenating their rows with unit weights does not reproduce the physical pT
spectrum. New Parquet contains `Generator.weight`, `Generator.binvar`, and event
identifiers. Combined-mode receipts contain the preselection generator-weight sums,
including events with no electron candidates. Keep those receipts beside the
Parquet files. The `postprocess.py` driver normalizes the QCD bins and the TT
extension group using the verified cross sections in `config/postprocess.yaml`;
see the main README. Training sample balance and physical yield weighting are
separate. A DAS cross-section field of zero is not a usable normalization.

## Checks when something fails

**`ModuleNotFoundError: PhysicsTools.NanoAOD.run3scouting_cff` after building:**
if `sys.path` contains LCG paths but no CMSSW package paths, the runtime is
contaminated. Running `cmsenv` in that shell may not remove the inherited
Python and library settings. First test the existing build in a clean process;
set `cmssw_area` to the same directory as `cmssw.base` in your YAML:

```bash
cmssw_area="$HOME/Scouting_EGM_ID_pipeline/CMSSW_15_1_0_pre3"
env -i HOME="$HOME" USER="$(id -un)" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
    /bin/bash --noprofile --norc -s -- "$cmssw_area" <<'CMSSW'
set -eo pipefail
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd "$1/src"
cmssw_runtime=$(scram runtime -sh)
eval "$cmssw_runtime"
python3 - <<'PY'
import os
import sys
from pathlib import Path
base = Path(os.environ["CMSSW_BASE"])
print("CMSSW_BASE:", base, flush=True)
print("Python executable:", sys.executable, flush=True)
expected = base / "src/PhysicsTools/NanoAOD/python/run3scouting_cff.py"
print("Source exists:", expected.is_file(), flush=True)
print("Python search paths:", sys.path, flush=True)
import PhysicsTools.NanoAOD as nano
print("NanoAOD package paths:", list(nano.__path__), flush=True)
import PhysicsTools.NanoAOD.run3scouting_cff as scouting
print("Loaded:", scouting.__file__)
assert Path(scouting.__file__).resolve() == expected.resolve()
assert hasattr(scouting, "scoutingElectronRegressVarTable")
print("CONFIGURATION OK")
PY
CMSSW
```

If this prints `CONFIGURATION OK`, rerun the main README's local test without
rebuilding. If the source is missing, check the checkout and package setup below.
If it still fails, keep the printed interpreter, search paths, and package paths
for diagnosis.

Do not require a direct `python/PhysicsTools/NanoAOD/run3scouting_cff.py` link.
[SCRAM's package initializer](https://github.com/cms-sw/cmssw-config/blob/scramv3/SCRAM/pyinit.py)
can extend the package search path to `src/PhysicsTools/NanoAOD/python` instead.
The earlier direct-link check was not a reliable test of build success.

**`Process` has no `scoutingElectronRegressVarTable`:** the custom Python
configuration was not loaded. In the configured CMSSW `src` directory after
`cmsenv`, check:

```bash
git status --short
git log -1 --oneline
python3 - <<'PY'
import PhysicsTools.NanoAOD.run3scouting_cff as scouting
print(scouting.__file__)
print("Regression producer available:", hasattr(scouting, "scoutingElectronRegressVarTable"))
PY
```

The reference revision defines this producer. A path under the standard release
on CVMFS usually means the local NanoAOD package is absent or not linked by the
build. A local path with `False` means that copy lacks the custom definition.
The earlier README used `git checkout --detach` before `git cms-addpkg`; CMS's
package script requires a named branch. If `HEAD` is already at the reference
revision, attach it to a new branch and finish setup:

```bash
git checkout -b codex/scouting-id-runtime
git cms-addpkg PhysicsTools/Scouting
git cms-addpkg PhysicsTools/NanoAOD
scram b -j 4
cmsenv
```

If `HEAD` differs, use the main README's pinned setup in a clean checkout.
Repeat the import check; it must print `True`. Then rerun the local test.
Updating the shared CMSSW area takes effect without regenerating the farm;
changes to this repository's `nano_cfg.py` require a new farm snapshot.

**Missing CMSSW plugin:** in your CMSSW `src` directory, run:

```bash
cmsenv
edmPluginHelp -p Run3ScoutingEGammaP4RegressVarProducer
edmPluginHelp -p Run3ScoutingElectronBestTrackProducer
```

If these fail, check out and compile the two packages listed in the main README.

**Coffea import failure:** in a fresh shell in this repository, run:

```bash
source env.sh
python3 -c 'import id_ntupler; import pyarrow.parquet; print("Converter imports OK")'
```

Use your own setup wrapper if `env.sh` does not provide the required dependencies.

**`NoRecord: PDTRecord` in `finalGenParticles`:** the MC generator-particle
pruner requires the particle data table. `nano_cfg.py` loads it with
`process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")`. No CMSSW rebuild is
needed. Farms contain a snapshot of this file: prepare a new farm after updating
the repository, then repeat the complete local test in a new temporary directory.
For example:

```bash
python3 ntuplizer.py config/ntuplizer.yaml \
    --farm-dir farms/id_smoke_pdt --samples DYToLL_M50 \
    --max-files 1 --max-events 100
```

In the main README's local-test block, set `farm="$PWD/farms/id_smoke_pdt"`.
The Winter25 DY test reached this error after its scouting electron, photon,
vertex, and rho modules succeeded on two events. This is partial input validation,
not a completed Nano-to-Parquet test. `TrigReport Events ... passed` alone is not
success when the output end path reports errors.

**Missing input products:** inspect one input file from the generated job JSON.
In the CMSSW environment, run `edmDumpEventContent` on that file's XRootD URL.
The input must contain `hltScoutingEgammaPacker`,
`hltScoutingPrimaryVertexPacker:primaryVtx`, `hltScoutingPFPacker:rho`, and the
MiniAOD MC gen/pileup products. Ordinary MiniAOD without scouting products is
not sufficient. The default recipe needs the 2024 scouting electron fields;
the local test of the previously configured Winter22 dataset failed on its first two
processed events with missing scouting electrons and `hltScoutingPFPacker:rho`.
Opening the ROOT file successfully only verifies access, not product compatibility.
Do not use `TryToContinue` to treat this test as successful: skipping those
events does not supply the electron features needed for ID training.

This pipeline consumes scouting products; it does not execute the scouting HLT
menu. Use compatible scouting MC, or first establish an upstream HLT production
with the required input tier and menu. Ordinary MiniAOD alone is not a general
substitute for the raw/digitized detector inputs that an HLT rerun may require.
After changing the sample, prepare a new farm and repeat the local test.

**Held job:** run `condor_q -hold` and inspect the farm's `logs/` directory.
Processing failures hold the job. Upload attempts are retried a limited number
of times; there is no processing checkpoint. Scratch is cleaned on exit, so a
retry repeats both stages. An empty electron table with a valid schema is retained:
its processed events must still contribute to the normalization denominator.
Use a larger smoke sample if there are no electrons to inspect.

**`Fatal system signal has occurred during exit` after the event summary:**
job `16467473.477` processed all 46,760 input events, then `cmsRun` terminated
with `SIGABRT`. The Python worker propagated this as exit code 1 before conversion
or upload. This identifies a shutdown crash, not its underlying cause. The
[CMSSW shutdown signal handler](https://github.com/cms-sw/cmssw/blob/CMSSW_15_1_0_pre3/FWCore/Services/plugins/InitRootHandlers.cc)
prints this message without a stack trace. A successful event summary alone
does not justify ignoring the process failure.

Reproduce the exact failed input using the farm snapshot, not a new DAS query or
the first file of the dataset. On lxplus, with a valid CMS proxy, run:

```bash
farm=/afs/cern.ch/user/t/tihsu/Scouting_EGM_ID_pipeline/Scouting_EGM_ID/farms/id_full
job_file=DYToLL_M50_00478.json
debug_dir=$(mktemp -d /tmp/scouting-debug.XXXXXX)
echo "Debug directory: $debug_dir"
export X509_USER_PROXY="$(voms-proxy-info -path)"
set -o pipefail
env -i HOME="$HOME" PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 \
    X509_USER_PROXY="$X509_USER_PROXY" \
    /bin/bash --noprofile --norc -s -- "$farm" "$job_file" "$debug_dir" \
    <<'BASH' 2>&1 | tee "$debug_dir/gdb.log"
set -eo pipefail
cp "$1/runtime.env" "$1/nano_cfg.py" "$3/"
cp "$1/jobs/$2" "$3/job.json"
cd "$3"
export TMPDIR="$PWD/tmp" XDG_CACHE_HOME="$PWD/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
source runtime.env
source "$CMSSET_DEFAULT"
cd "$CMSSW_AREA/src"
cmssw_runtime=$(scram runtime -sh)
eval "$cmssw_runtime"
test "$CMSSW_VERSION" = "$CMSSW_RELEASE"
cd "$3"
ulimit -c 0
command -v gdb
gdb --batch \
    -ex 'set pagination off' \
    -ex 'handle SIGPIPE nostop noprint pass' \
    -ex run -ex 'thread apply all bt' \
    --args cmsRun -j FrameworkJobReport.xml nano_cfg.py job=job.json
BASH
```

This runs Nano production only and retains logs and output in `debug_dir`; it
does not stage out to EOS. Keep the full `gdb.log`, especially the signal and
backtrace near its end. GDB's exit status is not a pipeline-success check. If
the inferior exits normally, the original failure was not reproduced in this
attempt. Do not assume all held jobs have this cause merely because they share
the same `OnExitHold` expression. Inspect representative logs before choosing a
recovery action, and do not bypass `cmsRun` exit-code checks.

**Output already exists:** use a new campaign for new processing settings.
A receipt-upload failure can leave a Parquet file without its JSON receipt;
inspect the corresponding held job before replacing anything.

## Validate the output

Require `COMPLETE` and exit code zero, then check all expected outputs listed in
`manifest.json`. Each Parquet contains one row per scouting electron, with
event-level values repeated for that event's electrons. Its JSON receipt records
input files, configuration, row count and conversion package versions.

Read the Parquet from its mounted EOS path with `pandas.read_parquet` and inspect
columns, row count, missing values and the `status` distribution. For a matched
comparison, run the same input file and event limit through combined and separate
production with identical settings, then compare the resulting tables allowing
for row-order differences.

## Electron truth labels

The Parquet `status` column is assigned in `id_ntupler.py`, separately from the
generator's `GenPart.status`. For each scouting electron, choose the nearest
final-state generator electron (`abs(pdgId) == 11`, `GenPart.status == 1`) in
the same event. Accept the match only when `deltaR <= gen_matching_dR`, set in
`workflow.yaml` (default 0.1). No generator electron or an out-of-cone match always
gives `status = 0`; ancestry cannot overwrite it. Matching is per reconstructed
electron and does not enforce a one-to-one assignment.

| Stored status | Definition | Selection within the configured MC samples |
| --- | --- | --- |
| `0` | Unmatched | Fake/background |
| `1` | Matched electron with `isPrompt` | Signal |
| `2` | Matched electron with `isDirectTauDecayProduct` | Signal |
| `3` | Other matched electron, including heavy-flavour decays | Fake/background |

Prompt and tau classification use the official
[CMS GenStatusFlags](https://github.com/cms-sw/cmssw/blob/CMSSW_15_1_0_pre3/DataFormats/HepMCCandidate/interface/GenStatusFlags.h),
bits 0 and 4. Direct tau decay takes precedence; no extra heavy-flavour-to-tau
ancestry rule is applied. Electrons with neither flag are category 3, which is
broader than heavy flavour alone. Missing parent records do not cause null labels
because the classification uses the electron's flags instead of parent indexing.

`config/standard.yaml` selects `(status == 1) | (status == 2)` for signal and
`(status == 0) | (status == 3)` for background, retaining `pt > 10` and the existing
sample lists. The effective-area selection uses the same signal categories.
These cuts are applied during analysis; the ntuplizer stores all four categories.

Old converter versions could overwrite unmatched labels and used a parent-based
prompt fallback. Existing Parquet cannot be repaired by changing the training
cuts alone: regenerate it from NanoAOD or MiniAOD using a new farm and a new EOS
campaign. Keep old and corrected outputs separate. Synthetic tests cover the
categories, empty collections, matching boundary, distant tau/hadron electrons,
missing parents, YAML cuts, and a processor-to-Parquet round trip. Check the
status counts and electron pT distributions on actual MC before training.

For analysis, set `inputdir` to the mounted EOS campaign directory and make the
`process` sample names match the produced directories. `training_input.yaml` is
only an `inputdir` fragment. The current analysis has `n_files: 4`; adjust it if
you intend to consume all files. An XRootD URL cannot replace a directory path
for the local/mounted Parquet consumer.

## Optional separate Nano production

Set `production.output_format: nano`. After the Nano jobs finish, use the
farm's generated `id_workflow.yaml` with the old converter:

```bash
python3 condor.py farms/nano_v1/id_workflow.yaml \
    --store_path /path/to/parquet_output --chunks 20 \
    --farm_dir farms/id_conversion --submit
```

Use your actual farm and output paths. The standalone converter retains its
existing storage design; the combined mode keeps both stages in worker scratch.

## Local tests

```bash
python3 -m unittest discover -s tests -v
bash -n ntuplizer/run.sh
```

These checks cover orchestration and local validation. They do not demonstrate
successful CMSSW/Coffea processing or EOS access on a CERN worker.
