# Scouting Egamma ID ntuplizer
## Workflow file format
To get proxy file path, run 
```
voms-proxy-init --rfc --voms cms -valid 192:00
```
Please replace the `proxy_file` in the workflow yaml file. The yaml file format is described below
```yaml
samples:
    process_name:
        path: [input_nanoaod_storage_path, AAA prefix not needed]
        fname: [file name format, index not needed]
        numbers: [total number of files]

gen_matching_dR: [float]

proxy_file: [proxy file storage place]

ntupler_store_values: [list of variables that want to store]
```
## Step1: GenMatching + Produce flat ntuple
To submit the jobs
```
python3 condor.py workflow.yaml --store_path [storage_path] --chunks 20 --farm_dir chunks --submit
```

