import os
import argparse
import math
import yaml

def split_files(all_files, num_chunks):
    chunk_size = math.ceil(len(all_files) / num_chunks)
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    return chunks

def write_chunk_lists(chunks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f"chunk_{i}.txt")
        with open(chunk_file, 'w') as f:
            f.write("\n".join(chunk))
        chunk_paths.append(chunk_file)
    return chunk_paths

def write_individual_run_script(script_name, chunk_file, store_path, index, workflow_file):
    with open(script_name, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        f.write(f"echo Running index {index} with chunk file {chunk_file}\n")
        f.write(f"cd {os.getcwd()}\n")
        f.write(f"source {os.getcwd()}/env.sh\n")
        # python3 id_ntupler.py workflow.yaml
        f.write(f"python3 id_ntupler.py {workflow_file} --store_path {store_path} --index {index} --files $(cat {chunk_file} | xargs)\n -c 3")
    os.chmod(script_name, 0o755)


def create_condor_submit(chunk_paths, store_path, out_file="submit.sub", run_dir="runs", workflow = "workflow.yaml", postfix = "", proxy_file = ""):
    os.makedirs("logs", exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    with open(out_file, "w") as f:
        f.write("universe = vanilla\n")
        f.write('+JobFlavour = "longlunch"\n')
        f.write("should_transfer_files = NO\n")
        f.write(f"x509userproxy = {proxy_file}\n")
        f.write("use_x509userproxy = True\n")
        f.write("getenv = True\n")
        f.write("request_cpus = 4\n")

        for i, chunk_file in enumerate(chunk_paths):
            run_script = f"{run_dir}/run_{i}.sh"
            write_individual_run_script(run_script, chunk_file, store_path, i, workflow)

            f.write(f"executable = {run_script}\n")
            f.write(f"output = logs/job{postfix}_{i}.out\n")
            f.write(f"error = logs/job{postfix}_{i}.err\n")
            f.write(f"log = logs/job{postfix}_{i}.log\n")
            f.write("queue\n\n")


def main():
    parser = argparse.ArgumentParser(description="Generate Condor job submission")
    parser.add_argument("workflow_yaml",  help="workflow yaml")
    parser.add_argument("--store_path", required=True, help="Path to the HDF5 store")
    parser.add_argument("--chunks", type=int, default=10, help="Number of chunks")
    parser.add_argument("--farm_dir", default="chunks", help="Directory to store chunk files")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    with open(args.workflow_yaml, "r") as f:
        config = yaml.safe_load(f)
    

    for process, info in config["samples"].items():
        store_path = info["path"]
        fname = info["fname"]
        number = info["numbers"]
        file_lists = ["root://cms-xrd-global.cern.ch//" + os.path.join(store_path,fname).replace(".root", f"_{i}.root") for i in range(1, number+1)]

        chunks = split_files(file_lists, args.chunks)

        chunks_dir = os.path.join(args.farm_dir, process)
        os.makedirs(chunks_dir, exist_ok = True)
        chunk_paths = write_chunk_lists(chunks, chunks_dir)
        os.makedirs("logs", exist_ok=True)
        output_store_path = os.path.join(args.store_path, process) 
        create_condor_submit(chunk_paths, output_store_path, workflow = args.workflow_yaml, run_dir = chunks_dir, out_file = os.path.join(chunks_dir, f"condor.submit"), postfix = process, proxy_file = config["proxy_file"])
        print(f"Prepared Condor submission for {len(chunk_paths)} jobs for {process}")

        if args.submit:
             os.system(f"condor_submit {chunks_dir}/condor.submit")

if __name__ == "__main__":
    main()

