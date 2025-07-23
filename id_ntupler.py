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

        # Add gen matching information
        electrons = events.ScoutingElectron
        gen_ele   = events.GenPart[(abs(events.GenPart.pdgId) == 11) & (events.GenPart.status == 1)]

        electrons["momentum"] = ak.zip({
            "pt": electrons.pt,
            "eta": electrons.eta,
            "phi": electrons.phi,
            "mass": electrons.m
        }, with_name = "Momentum4D")

        gen_ele["momentum"] = ak.zip({
           "pt": gen_ele.pt,
           "eta": gen_ele.eta,
           "phi": gen_ele.phi,
           "mass": gen_ele.mass
        }, with_name = "Momentum4D")


        ele_gen_pairs = ak.cartesian({"ele": electrons, "gen": gen_ele}, nested=True)
        dR = ele_gen_pairs["ele"]["momentum"].deltaR(ele_gen_pairs["gen"]["momentum"])

        # Find the closest gen electron per reco electron
        closest_idx = ak.argmin(dR, axis=2, keepdims=True)
        closest_dR = ak.min(dR, axis=2)
        closest_idx_flatten = ak.flatten(closest_idx, axis = -1)

        electrons["gen_idx"] = ak.flatten(closest_idx, axis = -1)
        electrons["gen_dR"] = closest_dR

        # status: 0: UNMATCHED, 1: MATCHED TO TRUE PROMPT, 2: MATCHED TO TRUE ELECTRON FROM TAU, 3: MATCHED TO NON-PROMPT
        electrons["status"] = ak.ones_like(electrons.pt)
        electrons["status"] = ak.where(electrons.gen_dR > self.gen_matching_dR, ak.zeros_like(electrons.status), electrons.status)

        matched_gen_ele = gen_ele[electrons.gen_idx]
        electrons["status"] = ak.where((abs(matched_gen_ele.distinctParent.pdgId) > 50) & (matched_gen_ele.status == 2), ak.ones_like(electrons.status)*3, electrons.status)
        electrons["status"] = ak.where((abs(matched_gen_ele.distinctParent.pdgId) == 15) & (matched_gen_ele.status == 2), ak.ones_like(electrons.status)*2, electrons.status)

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
    parser.add_argument("--cpu", default = None)
    # Parse command-line arguments
    args = parser.parse_args()
    with open(args.workflow_yaml, "r") as f:
        config = yaml.safe_load(f)

#    filename = "root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250506_TestSubmit/250506_101748/0000/scouting_nano_2.root"
#    filename = "root://cms-xrd-global.cern.ch///store/user/asahasra/ScoutingPFRun3/Scouting_2024F_crabNano250527_DiElSkim/250527_081921/0000/scouting_nano_100.root"

    n_workers = os.cpu_count() - 1 if args.cpu is None else args.cpu
    runner = Runner(
        executor=FuturesExecutor(compression=None, workers = n_workers),
        schema=NanoAODSchema,
        chunksize=15000,
        maxchunks=None,
    )

    valid_files = [f for f in args.files if is_valid_root_file_xrootd(f)]

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
    df = pd.DataFrame({k:v.value for k,v in output["columns"].items()})
    df.to_hdf(os.path.join(args.store_path, f"output_{args.index}.h5"), key="data", mode='w', format="table")

