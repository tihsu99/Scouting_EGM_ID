import awkward as ak
import numpy as np
import vector
import matplotlib.pyplot as plt
from coffea import processor
from coffea.processor import Runner, FuturesExecutor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
# Register vector methods
vector.register_awkward()
from hist import Hist
import subprocess
import re
import pandas as pd
import argparse
import os
import yaml
import uproot
from copy import deepcopy

def is_valid_root_file_xrootd(path):
    try:
        with uproot.open(path + ":Events") as _:
            return True
    except Exception:
        return False

def match_electron_truth(electrons, gen_particles, max_dr):
    """Label nearest final-state electrons: 0 unmatched, 1 prompt, 2 tau, 3 other."""
    gen_ele = gen_particles[(abs(gen_particles.pdgId) == 11) & (gen_particles.status == 1)]
    reco_momentum = ak.zip({
        "pt": electrons.pt, "eta": electrons.eta,
        "phi": electrons.phi, "mass": electrons.m,
    }, with_name="Momentum4D")
    gen_momentum = ak.zip({
        "pt": gen_ele.pt, "eta": gen_ele.eta,
        "phi": gen_ele.phi, "mass": gen_ele.mass,
    }, with_name="Momentum4D")
    pairs = ak.cartesian({"ele": reco_momentum, "gen": gen_momentum}, nested=True)
    dr = pairs.ele.deltaR(pairs.gen)
    closest_idx = ak.argmin(dr, axis=2)
    closest_dr = ak.min(dr, axis=2)
    matched = ak.fill_none(closest_dr <= max_dr, False)
    flags = ak.fill_none(gen_ele[closest_idx].statusFlags, 0)

    # CMS GenStatusFlags: bit 0 isPrompt, bit 4 isDirectTauDecayProduct.
    prompt = (flags & (1 << 0)) != 0
    from_tau = (flags & (1 << 4)) != 0
    category = ak.where(from_tau, 2, ak.where(prompt, 1, 3))
    electrons["status"] = ak.values_astype(ak.where(matched, category, 0), np.int32)
    electrons["gen_idx"] = closest_idx
    electrons["gen_dR"] = closest_dr
    return electrons


class Ntupler(processor.ProcessorABC):
    def __init__(self, config):
        self.config = config
        self.store_values_electron = config["ntupler_store_values_for_electron"]
        self.store_values_event = config["ntupler_store_values_for_event"]
        self.gen_matching_dR = config["gen_matching_dR"]

        self.store_values = deepcopy(self.store_values_electron)

        for collection, branches in self.store_values_event.items():
            for branch in branches:
                self.store_values.append(f"{collection}.{branch}")

        self._accumulator = processor.dict_accumulator({ 
            key: processor.column_accumulator(np.array([]))
            for key in self.store_values
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        electrons = match_electron_truth(events.ScoutingElectron, events.GenPart,
                                         self.gen_matching_dR)

        for key in self.store_values_electron:
            output[key] += processor.column_accumulator(ak.to_numpy(ak.flatten(electrons[key])))

        for collection, branches in self.store_values_event.items():
            obj = getattr(events, collection)
            for branch in branches:
                value = getattr(obj, branch)  # shape: (n_events,)
                # Broadcast to match electrons per event
                broadcasted = ak.broadcast_arrays(electrons.pt, value)[1]
                output[f"{collection}.{branch}"] += processor.column_accumulator(ak.to_numpy(ak.flatten(broadcasted)))


        return {
            "columns": output,
        }

    def postprocess(self, accumulator):
        pass
# === Run and plot ===
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Explore the structure of an HDF5 file")
    parser.add_argument("workflow_yaml", type=str)
    parser.add_argument("--store_path", help="Path to the HDF5 file")
    parser.add_argument("--index", default = 0)
    parser.add_argument("--files", nargs = "+")
    parser.add_argument("--cpu", type=int, default = None)
    parser.add_argument("--chunk-size", type=int, default=15000)
    parser.add_argument("--strict-inputs", action="store_true",
                        help="Fail if any input is unreadable instead of skipping it")
    # Parse command-line arguments
    args = parser.parse_args()
    if args.chunk_size < 1 or (args.cpu is not None and args.cpu < 1):
        parser.error("--chunk-size and --cpu must be positive")
    with open(args.workflow_yaml, "r") as f:
        config = yaml.safe_load(f)

#    filename = "root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250506_TestSubmit/250506_101748/0000/scouting_nano_2.root"
#    filename = "root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250527_DiElSkim/250527_081921/0000/scouting_nano_100.root"

    n_workers = os.cpu_count() - 1 if args.cpu is None else args.cpu
    runner = Runner(
        executor=FuturesExecutor(compression=None, workers = n_workers),
        schema=NanoAODSchema,
        chunksize=args.chunk_size,
        maxchunks=None,
    )

    valid_files = [f for f in args.files if is_valid_root_file_xrootd(f)]
    if args.strict_inputs and len(valid_files) != len(args.files):
        raise RuntimeError("Unreadable input ROOT files: " + ", ".join(f for f in args.files if f not in valid_files))
    if not valid_files:
        raise RuntimeError("No readable input ROOT files")

    fileset = {
        "mydataset": valid_files  # args.files should be a list of file paths
    }
    output = runner(
        fileset,
        treename="Events",
        processor_instance=Ntupler(config),
    )

    os.makedirs(args.store_path, exist_ok=True)

    for k, v in output["columns"].items():
      print(k, len(v.value))

    print(output["columns"]) 
    # Save filtered events (as NumPy or print summary)
    #df = pd.DataFrame({k:v.value for k,v in output["columns"].items()})
    #df.to_hdf(os.path.join(args.store_path, f"output_{args.index}.h5"), key="data", mode='w', format="table")


    df = pd.DataFrame({k: v.value for k, v in output["columns"].items()})
    output_path = os.path.join(args.store_path, f"output_{args.index}.parquet")
    df.to_parquet(output_path, index=False)

    print(f"[✓] Saved filtered events to: {output_path}")
