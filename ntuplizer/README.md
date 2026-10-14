# Scouting NanoAOD and ID training ntuple production

This stage reads scouting-enabled **MINIAODSIM** datasets from DAS. The default
`production.output_format: parquet` combines CMSSW Nano production and the
existing `id_ntupler.py` conversion in one Condor job. Only the final training
Parquet file and provenance receipt are stored on EOS. The intermediate Nano
`Events` tree stays in worker scratch. Set `output_format: nano` to retain the
separate Nano production flow; older configs without this setting keep that mode.

The Nano stage uses the scouting electron, photon, primary
vertex and rho tables, the regression-variable producer, and the standard Nano
generator/pileup tables. It does not apply an event skim or an electron ID cut.
The configured best-track selection still affects track association.

The supplied PDF describes a separate flat energy-regression analyzer. That tree
is not the input format of this repository's ID workflow. The Nano recipe here
uses the same scouting fork, with reference commit
`3bcde91ff6ce2e635c6672b5dfa3774a2a427986`, and schedules the Nano tables directly.
See the upstream [scouting table definitions](https://github.com/CMS-Run3ScoutingTools/cmssw/blob/3bcde91ff6ce2e635c6672b5dfa3774a2a427986/PhysicsTools/NanoAOD/python/run3scouting_cff.py).

The example keeps the upstream Nano best-track `TrackPtMin = [12, 12]` GeV;
the PDF's flat analyzer uses `[5, 5]`. This is an explicit YAML setting. The old
ROOT files' exact production settings have not been recovered, so equality with
their physics content must be checked on real events before replacing them.

## Storage and job lifecycle

1. The submit host queries DAS once and saves sorted, deduplicated LFNs in job
   JSON files. Alternatively, provide a saved `file_list` for each sample.
2. Condor transfers only the scripts, resolved job and small configuration files.
3. The worker requires `_CONDOR_SCRATCH_DIR`. It activates your existing compiled
   CMSSW area from `cmssw.base` using `scram runtime -sh`, then immediately returns
   to its private scratch directory. It does not copy, compile or relocate CMSSW.
4. CMSSW streams source ROOT files through XRootD. The output ROOT file and
   temporary/cache files stay in scratch. There is no full local input copy.
5. The worker checks that the output is nonempty, readable at its first and last
   events, and contains every input branch required by the saved ID workflow.
   `ScoutingElectron_status` is excluded because the ID code computes it.
6. In Parquet mode, a fresh subprocess sources `conversion.setup_script` and
   runs the snapshotted `id_ntupler.py`. It uses the branch list and
   `gen_matching_dR` from `production.id_workflow`, preserving the existing
   matching and status-label implementation. CMSSW library/Python paths are
   excluded from the conversion environment. Workers and chunk size come from
   YAML. The output must have the requested columns, at least one electron row,
   and readable first/last row groups before the local Nano is removed.
7. `xrdcp --cksum adler32` uploads to a unique partial name. Only a successful copy
   is renamed to `output_N.parquet` (or `scouting_nano_N.root` in Nano mode).
   A JSON provenance receipt follows it.
   Failed copies are retried a bounded number of times. Failures hold the job.
8. Scratch is cleaned on normal shell exit, including errors. Condor returns
   stdout/stderr, not ROOT files or the CMSSW area, to the submit host.

This follows HTCondor's [scratch and explicit file-transfer model](https://htcondor.readthedocs.io/en/latest/users-manual/file-transfer.html).
`disk_mb` must cover the Nano and Parquet outputs during conversion, temporary
files and caches, with headroom. The shared CMSSW area is not copied to scratch.
Worker scratch is finite: start with one file per job and measure it.
EOS quota and proxy lifetime must also cover the production.
The existing converter accumulates a job's electron rows in memory; smaller
`chunk_size` alone does not bound that final table. Use `files_per_job` and
`memory_mb` accordingly. A conversion failure requires rerunning CMSSW as well.

## 1. Select your existing CMSSW area

Do this in a compatible CMS EL9 environment with CVMFS and XRootD, not on macOS.
On an EL9 host, use its normal shell; there is no need to enter `cmssw-el8`.
The [CMS release map](https://github.com/cms-sw/cms-bot/blob/master/releases.map)
lists `CMSSW_15_1_0_pre3` for `el9_amd64_gcc12`. `condor.os` and
`cmssw.scram_arch` must match your compiled runtime. The default `+WantOS = "el9"`
is the CERN batch setting; adapt the submit template for a non-CERN pool.
You maintain and compile the CMSSW area in your home/shared filesystem. Set
`cmssw.base` to its release directory, not its `src` directory. The path is
expanded on the submission host and saved as an absolute path in each job.
That same path must be readable from the Condor worker. A directory that exists
only on the submit machine's local disk is not sufficient.

For example, check your already compiled area in a separate shell:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd "$HOME/CMSSW_15_1_0_pre3/src"  # Use your actual maintained area.
eval "$(scram runtime -sh)"
echo "$CMSSW_BASE $CMSSW_VERSION $SCRAM_ARCH"
git rev-parse HEAD
edmPluginHelp -p Run3ScoutingEGammaP4RegressVarProducer >/dev/null
```

The area must contain the required `PhysicsTools/Scouting` plugin and modified
`PhysicsTools/NanoAOD` configuration. Preparation checks for `src`; the worker
checks accessibility and the activated release. The plugin check and smoke job
verify more than the existence of a directory. Keep the shared area unchanged
while jobs are queued or running; they use its live libraries and Python files.
The YAML source commit is a declared reference, not a verified snapshot of that
shared area. If moving from EL8, compile your maintained area for EL9 first.

## 2. Configure samples and prepare a smoke job

Return to the Scouting_EGM_ID checkout. The submission Python needs PyYAML;
`source env.sh` provides the repository's normal LCG environment. Nano-only
workers use CMSSW's Python and ROOT. Combined workers also need an EL9-compatible
Coffea environment with the existing converter's APIs and dependencies.

`conversion.setup_script` defaults to the repository's `env.sh`. This wrapper is
copied into the farm and onto workers; it must use absolute paths for everything
it sources. You can point it to your own self-contained LCG/venv setup wrapper.
The wrapper must make `conversion.python` available without relying on an
interactive shell or inherited CMSSW variables. Do not put credentials in it.
The repository's LCG environment has not been exercised here on a CERN worker;
verify it before submitting, from a fresh shell without CMSSW activated:

```bash
source env.sh  # Or the wrapper selected by conversion.setup_script.
python3 -c 'import id_ntupler; import pyarrow.parquet; print("ID converter imports OK")'
```

```bash
source env.sh
voms-proxy-init --rfc --voms cms -valid 192:00
export X509_USER_PROXY="$(voms-proxy-info -path)"
cp config/ntuplizer.yaml config/ntuplizer.local.yaml
```

Edit `config/ntuplizer.local.yaml`:

- Set `cmssw.base` to your existing compiled CMSSW area, accessible to workers.
- Set `production.eos_output` to your writable **full XRootD URL**, for example
  `root://eosuser.cern.ch//eos/user/u/username/scouting_id`.
- Use `production.campaign: id_parquet_smoke_v1` for the smoke run.
- Keep `production.output_format: parquet` for direct training input. Set
  `conversion.setup_script`, `python`, `workers` and `chunk_size` for your
  validated Coffea environment. Conversion workers cannot exceed `condor.cpus`.
- Enter full three-component DAS names under `samples`. Add any number of samples,
  use `enabled: false` to skip one, and optionally override `files_per_job` per sample.
- Check the modifier against the input format; the supplied workflow requests
  the 2024 scouting electron variables. This configuration is not a raw-data
  reconstruction chain. Datasets must actually contain the scouting products.
- Adjust disk, memory, CPUs, job flavour, and best-track thresholds as needed.
  The `pset` path can select a different CMSSW configuration without editing the
  submission code; it must honor the job JSON and `scouting_nano.root` contract.

Paths inside YAML are relative to that YAML file; environment variables expand
for local paths such as the proxy. EOS URLs must be explicit. Existing farm
directories and unresolved placeholders are rejected.

```bash
python3 ntuplizer.py config/ntuplizer.local.yaml \
    --farm-dir farms/id_smoke --samples DYToLL_M50 \
    --max-files 1 --max-events 100

cat farms/id_smoke/submit.sub
cat farms/id_smoke/jobs/DYToLL_M50_00001.json
condor_submit -dry-run farms/id_smoke/submit.ads farms/id_smoke/submit.sub
condor_submit farms/id_smoke/submit.sub
condor_q
```

Preparation alone does not submit. `--submit` runs a Condor dry-run first and then
submits. Without `file_list`, preparation needs a working `dasgoclient` and CMS
credentials. A DAS failure or empty file list aborts the entire preparation.
No submitted jobs are silently omitted.

## 3. Check the smoke result, then run all events

Require a `COMPLETE` line in the job stdout and exit code 0 in the Condor log.
For a held job use `condor_q -hold` and inspect its stdout/stderr. Typical causes
include an inaccessible/incompatible CMSSW area, absent scouting products, expired proxy,
insufficient disk, an unwritable EOS directory, or a missing required branch.

Read the output URL from `farms/id_smoke/manifest.json`, check it using `xrdfs`,
and inspect the produced Parquet in the EOS-mounted directory before launching
the full sample. Check the columns and status distribution. Compare a smoke
sample with the previous two-step production before replacing training inputs.

```bash
# Replace this with the mounted path corresponding to the manifest output URL.
parquet_path=/eos/user/CHANGE_ME/scouting_id/id_parquet_smoke_v1/DYToLL_M50/output_1.parquet
python3 - "$parquet_path" <<'PY'
import sys
import pandas as pd
frame = pd.read_parquet(sys.argv[1])
print(frame.shape)
print(frame.columns.tolist())
print(frame['status'].value_counts(dropna=False))
print(frame[['pt', 'eta', 'Pileup.nPU']].head())
PY
```

For full production, change the YAML campaign to a **new** value such as
`id_parquet_v1`, use a new farm, and omit both smoke limits:

```bash
python3 ntuplizer.py config/ntuplizer.local.yaml \
    --farm-dir farms/id_parquet_v1 --submit
```

Do not edit the generated farm while its jobs are queued/running. A job's saved
JSON is its active configuration, including event limits and physics settings.
The shared CMSSW area remains external to the farm and is not snapshotted.

Each output has a sibling `.parquet.json` (or `.root.json`) receipt with input LFNs/URLs, DAS dataset,
CMSSW area path and declared settings, physics settings, required branches, event count and size.
Parquet receipts also include row count, columns, ID configuration, converter and
environment-wrapper hashes, and the Coffea/Awkward/Uproot/Pandas/PyArrow versions.
Check all expected outputs against `manifest.json` before the next stage. The
generated YAML files are not completion certificates.

Completed output names are not overwritten. Use a new campaign for a rerun with
changed settings or input lists. A failure after the output upload but before the
receipt upload can leave a file without a receipt; inspect that held job
before removing/replacing anything. Killing a worker abruptly may leave a uniquely
named `.partial-*` file on EOS; it is never listed in the downstream workflow.

## 4. Hand off to the existing ID pipeline

In **Parquet mode**, skip `condor.py` and go directly to analysis/training. Files
have the layout expected by `config/standard.yaml`:

```text
<eos_output>/<campaign>/<sample>/output_1.parquet
<eos_output>/<campaign>/<sample>/output_2.parquet
```

After **all** jobs pass, copy the `inputdir` value from
`farms/id_parquet_v1/training_input.yaml` into your analysis YAML. This generated
file is a path fragment, not a complete analysis configuration. Check that the
analysis `process` sample names match the production samples; keep your existing
weights, cuts and training settings. The consumer uses `os.listdir`, so run it
where EOS is mounted or copy the sample directories locally and set `inputdir`
to that local directory. A `root://` URL is not a directory path for this consumer.

```bash
python3 analysis.py --config-path config/standard.yaml --local-scheduler
```

In **Nano mode**, the generated `id_workflow.yaml` retains the ID branch list
and matching settings and replaces `samples` with explicit EOS URLs. The old
`path`/`fname`/`numbers` input style still works.

```bash
python3 condor.py farms/nano_v1/id_workflow.yaml \
    --store_path /path/to/parquet_output --chunks 20 \
    --farm_dir farms/id_v1 --submit
```

Use the actual farm path in the command above. The scratch/EOS protections cover
both stages in combined mode. When running `condor.py` separately, that existing
Parquet stage retains its own storage design. Its wrapper now forwards
explicit input URLs and a valid integer CPU count.

Data production is deliberately rejected by this initial ID recipe: the existing
consumer unconditionally needs MC `GenPart` and `Pileup` for truth labels. A data
mode needs a corresponding change in that consumer before it can be supported.

## Local validation

```bash
python3 -m unittest discover -s tests -v
bash -n ntuplizer/run.sh
git diff --check
```

These tests cover planning, failure paths, transfer commands, environment
isolation, real Parquet schema checks (when PyArrow is installed), and the handoff.
They do not demonstrate that CMSSW executes successfully on a real CMS dataset,
that your home CMSSW area is accessible on a worker, or that CERN/EOS accepts a production job.
The smoke run above is the required runtime check.
