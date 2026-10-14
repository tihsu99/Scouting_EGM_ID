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

## Checks when something fails

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

**Missing input products:** inspect one input file from the generated job JSON.
In the CMSSW environment, run `edmDumpEventContent` on that file's XRootD URL.
The input must contain `hltScoutingEgammaPacker`,
`hltScoutingPrimaryVertexPacker:primaryVtx`, `hltScoutingPFPacker:rho`, and the
MiniAOD MC gen/pileup products. Ordinary MiniAOD without scouting products is
not sufficient. The default recipe needs the 2024 scouting electron fields;
the currently configured Winter22 dataset is not validated for this recipe.

**Held job:** run `condor_q -hold` and inspect the farm's `logs/` directory.
Processing failures hold the job. Upload attempts are retried a limited number
of times; there is no processing checkpoint. Scratch is cleaned on exit, so a
retry repeats both stages. An empty electron table is currently treated as a
failure.

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

The existing converter can overwrite `status = 0` for an out-of-cone match with a
parent-based `status = 2` or `3`. This issue remains unresolved. Its default
`status = 1` is not an explicit prompt-flag test. Validate these semantics before
using the labels for training.

For analysis, set `inputdir` to the mounted EOS campaign directory and make the
`process` sample names match the produced directories. `training_input.yaml` is
only an `inputdir` fragment. The current analysis has `n_files: 4`; adjust it if
you intend to consume all files. An XRootD URL cannot replace a directory path
for the existing `os.listdir` consumer.

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
