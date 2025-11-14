import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib.pyplot as plt
import mplhep as hep
from correctionlib import highlevel
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.table import Table
console = Console()

plt.style.use(hep.style.CMS)

def apply_corrections(df, corrfile="corrections_all.json"):
    cs = highlevel.CorrectionSet.from_file(corrfile)
    for var_name in cs.keys():  # automatically iterate over all stored corrections
        corr = cs[var_name]
        df[var_name] = df["eta"].apply(lambda e: corr.evaluate(abs(e)))
    return df
# ---------------------
# Top-level functions
# ---------------------

def build_hist_2d(args, requirement, pt_bins, eta_bins):
    """Build 2D histogram from a single file."""
    file_path, weight = args
    df = pd.read_parquet(file_path)
    if requirement:
        df = df.query(requirement)

    h, _, _ = np.histogram2d(
        df["pt"].values,
        df["eta"].values,
        bins=[pt_bins, eta_bins],
        weights=np.full(len(df), weight)
    )
    return h


def process_file(args, variables, fileset, reweight_2d=None, pt_bins=None, eta_bins=None, extra_requirement=None, correction_lib=None, variable_definition=None):
    """Process a single file, optionally applying 2D reweighting."""
    file_path, sample_name = args
    df = pd.read_parquet(file_path)
    if correction_lib is not None:
      df = apply_corrections(df, correction_lib)
    if variable_definition is not None:
      for variable_name, definition in variable_definition.items():
          df[variable_name] = df.eval(definition)
    requirement = fileset[sample_name]["requirement"]
    if requirement:
        df = df.query(requirement)
    if extra_requirement:
        df = df.query(extra_requirement)
    weights = np.ones(len(df)) * fileset[sample_name]["weight"]
    # Apply reweight if provided
    if reweight_2d is not None:
        pt_idx = np.clip(np.digitize(df["pt"].values, pt_bins) - 1, 0, len(pt_bins)-2)
        eta_idx = np.clip(np.digitize(df["eta"].values, eta_bins) - 1, 0, len(eta_bins)-2)
        weights *= reweight_2d[pt_idx, eta_idx]
    df["weight"] = weights


    out = {}
    for var, bins in variables.items():
        if len(bins) == 3:
            bins = np.linspace(bins[0], bins[1], bins[2]+1)
        if var in df.columns:
            out[var] = np.histogram(df[var].values, bins=bins, weights=weights, density = True)[0]
    out["df"] = df
    return out


def build_reference_hist(files, weights, requirement, pt_bins, eta_bins, n_workers=8):
    """Build 2D histogram (sum over multiple files) using multiple CPUs."""
    tasks = list(zip(files, weights))
    h2d_total = np.zeros((len(pt_bins)-1, len(eta_bins)-1))

    worker = partial(build_hist_2d, requirement=requirement, pt_bins=pt_bins, eta_bins=eta_bins)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for h in executor.map(worker, tasks):
            h2d_total += h

    return h2d_total

def split_and_save(signal_df, background_df, storedir, region_name, train_ratio=0.8, shuffle=True):
    # --- Split ---
    sig_train, sig_valid = train_test_split(
        signal_df, test_size=1 - train_ratio, random_state=42, shuffle=shuffle
    )
    bkg_train, bkg_valid = train_test_split(
        background_df, test_size=1 - train_ratio, random_state=42, shuffle=shuffle
    )

    # --- Prepare output dirs ---
    train_store_dir = os.path.join(storedir, region_name, "train")
    valid_store_dir = os.path.join(storedir, region_name, "valid")
    os.makedirs(train_store_dir, exist_ok=True)
    os.makedirs(valid_store_dir, exist_ok=True)

    # --- Save ---
    sig_train.to_parquet(os.path.join(train_store_dir, "signal.parquet"))
    sig_valid.to_parquet(os.path.join(valid_store_dir, "signal.parquet"))
    bkg_train.to_parquet(os.path.join(train_store_dir, "background.parquet"))
    bkg_valid.to_parquet(os.path.join(valid_store_dir, "background.parquet"))

    # --- Pretty summary ---
    table = Table(title=f"📊 Dataset Split Summary for [bold cyan]{region_name}[/bold cyan]")

    table.add_column("Sample", justify="center", style="bold")
    table.add_column("Train", justify="right", style="green")
    table.add_column("Valid", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="bold white")
    table.add_column("Train %", justify="right", style="cyan")

    def add_row(name, train_df, valid_df):
        n_train = len(train_df)
        n_valid = len(valid_df)
        n_total = n_train + n_valid
        frac_train = 100 * n_train / n_total if n_total > 0 else 0
        table.add_row(
            name,
            f"{n_train:,}",
            f"{n_valid:,}",
            f"{n_total:,}",
            f"{frac_train:.1f}%"
        )

    add_row("Signal", sig_train, sig_valid)
    add_row("Background", bkg_train, bkg_valid)

    console.print(table)
    console.print(f"💾 Saved to: [bold green]{storedir}/{region_name}/[/bold green]\n")



# ---------------------
# Main Analysis
# ---------------------

def InspectDistributionAnalysis(**kwargs):
    """
    Main analysis function for InspectDistribution workflow.

    kwargs must include:
        inputdir, process, kinematic_reweighting, plotdir, inspect_variable, n_workers
    """
    inputdir = kwargs.pop("inputdir")
    process = kwargs.pop("process")
    plotdir = kwargs.pop("plotdir")
    inspect_variable = kwargs.pop("inspect_variable")
    kinematic_reweighting = kwargs.pop("kinematic_reweighting", None)
    n_workers = kwargs.pop("n_workers", 8)
    n_files = kwargs.pop("n_files", -1)
    regions = kwargs.pop("regions", [])
    region_definition = kwargs.pop("region_definition", {})
    correction_lib = kwargs.pop("correction_lib", None)
    variable_definition = kwargs.pop("define_variable", None)
    storedir = kwargs.pop("storedir")
    shuffle = kwargs.pop("shuffle", True)
    train_ratio = kwargs.pop("train_ratio", 0.8)


    region_ref = {region_name: region_dict for region_name, region_dict in region_definition.items() if region_name in regions}

    os.makedirs(plotdir, exist_ok=True)
    print(f"✅ [InspectDistribution] plot dir: {plotdir}")
    # --- Build fileset ---
    fileset = {}

    for category, cat_info in process.items():
        req = cat_info.get("requirement", None)  # e.g., "status == 1"
        samples = cat_info["samples"]
        for sample_name, weight in samples.items():
            sample_path = os.path.join(inputdir, sample_name)
            files = [os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith(".parquet")]
            if n_files > 0:
                n_files_sample = min(n_files, len(files))
                files = files[:n_files_sample]


            fileset[sample_name] = {
                "files": files,
                "weight": weight,
                "requirement": req
            }

    # --- Build pt vs eta histograms for signal and background ---
    pt_bins = np.linspace(0, 200, 201)
    eta_bins = np.linspace(-3, 3, 51)

    # Background histogram
    bkg_files = []
    bkg_weights = []
    for sample_name, weight in process.get("background", {}).get("samples", {}).items():
        bkg_files.extend(fileset[sample_name]["files"])
        bkg_weights.extend([weight]*len(fileset[sample_name]["files"]))
    bkg_requirement = process.get("background", {}).get("requirement", None)
    H_bkg = build_reference_hist(bkg_files, bkg_weights, bkg_requirement, pt_bins, eta_bins, n_workers=n_workers)

    # Signal histogram
    sig_files = []
    sig_weights = []
    for sample_name, weight in process.get("signal", {}).get("samples", {}).items():
        sig_files.extend(fileset[sample_name]["files"])
        sig_weights.extend([weight]*len(fileset[sample_name]["files"]))
    sig_requirement = process.get("signal", {}).get("requirement", None)
    H_sig = build_reference_hist(sig_files, sig_weights, sig_requirement, pt_bins, eta_bins, n_workers=n_workers)

    # --- Compute reweight factor: w = H_bkg / H_sig ---

    reweight_2d = {"signal": None, "background": None}
    if kinematic_reweighting.lower() == "background":
        reweight_2d["signal"] = np.divide(H_bkg, H_sig, out=np.ones_like(H_bkg), where=H_sig>0)
    if kinematic_reweighting.lower() == "signal":
        reweight_2d["background"] =  np.divide(H_sig, H_bkg, out=np.ones_like(H_sig), where=H_bkg>0)


    for region_name, region_info in region_ref.items():
        os.makedirs(os.path.join(plotdir, region_name), exist_ok=True)

        # --- Process all files ---
        all_results = {"signal": {var: [] for var in inspect_variable},
                       "background": {var: [] for var in inspect_variable}}

        tasks_sig = [(f, s) for s in fileset if s in process.get("signal", {}).get("samples", {}) for f in fileset[s]["files"]]
        tasks_bkg = [(f, s) for s in fileset if s in process.get("background", {}).get("samples", {}) for f in fileset[s]["files"]]

        # Signal processing with reweight
        worker_sig = partial(process_file, variables=inspect_variable,
                             fileset=fileset, reweight_2d=reweight_2d["signal"],
                             pt_bins=pt_bins, eta_bins=eta_bins,
                             extra_requirement=region_info["requirement"],
                             correction_lib = correction_lib, variable_definition=variable_definition)
        # Background processing without reweight
        worker_bkg = partial(process_file, variables=inspect_variable,
                             fileset=fileset, reweight_2d=reweight_2d["background"],
                             pt_bins=pt_bins, eta_bins=eta_bins,
                             extra_requirement=region_info["requirement"],
                             correction_lib = correction_lib, variable_definition=variable_definition)

        signal_df = []
        background_df = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for res in executor.map(worker_sig, tasks_sig):
                for var in res:
                    if var == "df": 
                        continue
                    all_results["signal"][var].append(res[var])
                signal_df.append(res["df"])
            for res in executor.map(worker_bkg, tasks_bkg):
                for var in res:
                    if var == "df": 
                        continue

                    all_results["background"][var].append(res[var])
                background_df.append(res["df"])

        signal_df = pd.concat(signal_df, axis = 0)
        background_df = pd.concat(background_df, axis = 0)

        split_and_save(signal_df, background_df, storedir, region_name, train_ratio=train_ratio, shuffle=shuffle)



        # --- Sum histograms per variable ---
        final_hist = {}
        for var, bins in inspect_variable.items():
            final_hist[var] = {
                "signal": np.sum(all_results["signal"][var], axis=0),
                "background": np.sum(all_results["background"][var], axis=0)
            }

        colors = {
            "background": "#0072B2",  # color-blind friendly blue
            "signal": "#D55E00"       # color-blind friendly red
        }


        # --- Plot overlay: signal vs background ---
        for var, bins in inspect_variable.items():
            if len(bins) == 3:
                bins = np.linspace(bins[0], bins[1], bins[2]+1)
            plt.figure(figsize=(10, 5))
            hep.cms.text("Preliminary")
            hep.cms.label(data=True, lumi=None, year=2025)

            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            width = bins[1] - bins[0]

            # Signal histogram
            plt.bar(
                bin_centers, final_hist[var]["signal"],
                width=width, align='center',
                color=colors["signal"], alpha=0.5,
                edgecolor=colors["signal"], label="Signal (reweighted)", linewidth=1.2,
            )

            # Background histogram
            plt.bar(
                bin_centers, final_hist[var]["background"],
                width=width, align='center',
                color=colors["background"], alpha=0.5,
                edgecolor=colors["background"], label="Background", linewidth=1.2,
            )



            plt.text(
                0.05, 0.90,  # (x, y) position in axes fraction coordinates
                region_name.replace("_", " "),  # nice formatting
                transform=plt.gca().transAxes,  # interpret coords in axes space
                fontsize=18,
                fontweight="bold",
                ha="left",
                va="top",
            )

            plt.xlabel(var)
            plt.ylabel("Density")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(plotdir, region_name, f"{var}_density.pdf"))
            plt.yscale("log")
            plt.savefig(os.path.join(plotdir, region_name, f"{var}_density_log.pdf"))
            plt.close()
