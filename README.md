# Scouting Egamma ID ntuplizer
## Workflow file format
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
python3 condor.py workflow.yaml --store_path /eos/user/t/tihsu/public/ScoutingID_ntuple --chunks 20 --farm_dir chunks --submit
```

