#!/usr/bin/env python3
import os, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from workflow.plotting import add_legend, configure_style, save_plot
from workflow.working_points import normalize_fixed_cuts, pass_fixed_cuts, load_configured_wps, print_id_table
from workflow.cut_plots import plot_cut_distribution
from workflow.sampling import sample_training_candidates
import xml.etree.ElementTree as ET

import ROOT
ROOT.gROOT.SetBatch(True)
from ROOT import TMVA, TCut

import ROOT, tempfile, os
from array import array

def rdf_to_ttree(rdf, treename="tree_tmp"):
    """Snapshot an RDataFrame to a temporary ROOT file and return the TTree."""
    tmpfile = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
    tmpfile.close()  # ROOT will write to this path
    print(tmpfile.name)
    rdf.Snapshot(treename, tmpfile.name, list(rdf.GetColumnNames()))
    tf = ROOT.TFile.Open(tmpfile.name)
    tree = tf.Get(str(treename))
    tf.Close()
    return tree, tmpfile.name  # caller can later os.remove(tmpfile.name)

# ---------------- Utilities ----------------
def ensure_dir(d): os.makedirs(d, exist_ok=True)

def weighted_efficiency(mask, weights):
    if len(mask) == 0: return 0.0
    wsum = np.sum(weights)
    if wsum <= 0: return 0.0
    return float(np.sum(weights[mask]) / wsum)

def passes(df, variables, cuts_one_sided, dirs):
    """cuts_one_sided: {var: thr or np.nan}, applied one-sided according to dirs"""
    if len(df) == 0: return np.zeros(0, dtype=bool)
    conds = []
    for v in variables:
        thr = cuts_one_sided.get(v, np.nan)
        x = df[v].values
        if np.isfinite(thr):
            if dirs[v] == "less":
                conds.append(x < thr)
            else:
                conds.append(x > thr)
        else:
            conds.append(np.ones_like(x, dtype=bool))
    return np.logical_and.reduce(conds) if conds else np.ones(len(df), dtype=bool)

def parse_tmva_wps(weights_xml_path):
    """Read populated MethodCuts bins; negative effB marks an unfilled bin."""
    root = ET.parse(weights_xml_path).getroot()
    variables = sorted(root.findall("./Variables/Variable"), key=lambda v: int(v.get("VarIndex")))
    result = []
    for block in root.findall("./Weights/Bin"):
        if float(block.get("effB")) < 0:
            continue
        cuts = block.find("Cuts")
        windows = {var.get("Expression"): (float(cuts.get(f"cutMin_{i}")), float(cuts.get(f"cutMax_{i}")))
                   for i, var in enumerate(variables)}
        if not windows or not np.isfinite(list(windows.values())).all():
            raise ValueError(f"Invalid TMVA cut windows in {weights_xml_path}")
        result.append({"eff": float(block.get("effS")), "window_cuts": windows})
    if not result:
        raise ValueError(f"No populated TMVA efficiency bins in {weights_xml_path}")
    return result


def pick_nearest_eff(wps, target_eff):
    if not wps: return None
    return min(wps, key=lambda d: abs(d["eff"] - target_eff))

def to_one_sided(window_cuts, directions):
    """Convert TMVA (min,max) windows to one-sided thresholds matching 'direction'."""
    one = {}
    for v, (vmin, vmax) in window_cuts.items():
        if directions[v] == "less":
            one[v] = vmax if np.isfinite(vmax) else np.nan
        else:
            one[v] = vmin if np.isfinite(vmin) else np.nan
    return one

# ---------------- TMVA Training for one WP with bounds ----------------
def tmva_train_one_wp_inmem(sig_df, bkg_df, var_list, bounds, out_dir, weight_col,
                            train_ratio, method_name, ncycles, popsize, random_seed, eff_target, n_signal, n_background, n_workers=1, efficiency_tolerance=0.03,
                            n_train_signal=None, n_train_background=None):
    """
    sig_df, bkg_df: pandas DataFrames
    bounds: {var: (lo, hi)} -> will be passed to CutsGA as CutRangeMin/Max
    Returns path to weights xml.
    """
    ensure_dir(out_dir)
    #ROOT.EnableImplicitMT()  # multi-CPU (OpenMP inside TMVA)
    ROOT.TMVA.gConfig().fOMPThreads = n_workers
    ROOT.gRandom.SetSeed(random_seed)

    outfile_path = os.path.join(out_dir, f"TMVA_{method_name}.root")
    output = ROOT.TFile(outfile_path, "RECREATE")

    factory = TMVA.Factory(
        "TMVAClassification",
        output,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I"
    )

    dataset_name = f"dataset_{method_name}"
    ROOT.TMVA.gConfig().GetIONames().fWeightFileDirPrefix = os.path.abspath(out_dir)
    dl = TMVA.DataLoader(dataset_name)

    var_list_extend = [v for v in var_list]
    if weight_col is not None:
        var_list_extend.append(weight_col)
    for v in var_list:
        dl.AddVariable(v, "F")

    # Convert pandas→RDataFrame→TTree (in memory, no .root saved)
    from_numpy = getattr(ROOT.RDF, "FromNumpy", None) or ROOT.RDF.MakeNumpyDataFrame
    sig_rdf = from_numpy({c: sig_df[c].to_numpy() for c in var_list_extend})
    bkg_rdf = from_numpy({c: bkg_df[c].to_numpy() for c in var_list_extend})
    sigTree, sig_path = rdf_to_ttree(sig_rdf, "sigTree")
    bkgTree, bkg_path = rdf_to_ttree(bkg_rdf, "bkgTree")
    sigFile = ROOT.TFile.Open(sig_path)
    bkgFile = ROOT.TFile.Open(bkg_path)
    sigTree = sigFile.Get("sigTree")
    bkgTree = bkgFile.Get("bkgTree")
    print(type(sigTree))

    dl.AddSignalTree(sigTree, 1.0)
    dl.AddBackgroundTree(bkgTree, 1.0)
    if weight_col is not None:
        dl.SetSignalWeightExpression(str(weight_col))
        dl.SetBackgroundWeightExpression(str(weight_col))

    n_sig_total = sig_df.shape[0]
    n_bkg_total = bkg_df.shape[0]

    n_sig_total = min(n_sig_total, n_signal)
    n_bkg_total = min(n_bkg_total, n_background)
    n_train_sig = int(n_sig_total * train_ratio) if n_train_signal is None else n_train_signal
    n_train_bkg = int(n_bkg_total * train_ratio) if n_train_background is None else n_train_background
    n_test_sig  = n_sig_total - n_train_sig
    n_test_bkg  = n_bkg_total - n_train_bkg


    dl.PrepareTrainingAndTestTree(TCut(""), TCut(""),
        f"nTrain_Signal={n_train_sig}:nTrain_Background={n_train_bkg}:nTest_Signal={n_test_sig}:nTest_Background={n_test_bkg}:SplitMode=Random:SplitSeed={random_seed}:!V")

    # Build CutsGA options with per-variable range constraints.
    name_opts, idx_opts = [], []
    for i, v in enumerate(var_list):
        lo, hi = bounds[v]
#        if np.isfinite(lo): 
#            name_opts.append(f"CutRangeMin[{v}]={lo}")
#            idx_opts.append(f"CutRangeMin[{i}]={lo}:VarProp[{i}]=FMin")
        if np.isfinite(hi): 
            name_opts.append(f"CutRangeMax[{v}]={hi}")
            idx_opts.append(f"CutRangeMax[{i}]={hi}:VarProp[{i}]=FMin")

    opts = [
        "!H","!V",
        f"Cycles={ncycles}",
        f"PopSize={popsize}",
        "SC_steps=10","SC_rate=5","SC_factor=0.95",
    ] + idx_opts

    print(f"[options] {opts}")

    factory.BookMethod(dl, TMVA.Types.kCuts, method_name, ":".join(opts))
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    # Optional cleanup
    sigFile.Close()
    bkgFile.Close()
    os.remove(sig_path)
    os.remove(bkg_path)
    weights_xml = os.path.join(os.path.abspath(out_dir), dataset_name, "weights", f"TMVAClassification_{method_name}.weights.xml")
    if not os.path.exists(weights_xml):
        raise FileNotFoundError(weights_xml)
    selected = pick_nearest_eff(parse_tmva_wps(weights_xml), eff_target)
    output.Close()
    if abs(selected["eff"] - eff_target) > efficiency_tolerance:
        raise ValueError(f"TMVA did not populate a bin near efficiency {eff_target}; increase cycles/population or statistics")
    return weights_xml, selected["window_cuts"], selected["eff"]


# ---------------- Full sequential-bound pipeline ----------------
def train_cutbased_tmva_sequential_inmem(
    train_signal_path, train_bkg_path, valid_signal_path, valid_bkg_path,
    variables, target_efficiencies, outdir, train_ratio=0.8, weight_col="weight",
    tmva_cycles=60, tmva_popsize=120, random_seed=12345, tag="Default", n_workers = 1, n_signal = 9e19, n_background=9e19, isolation_denominator="unspecified", efficiency_tolerance=0.03,
    plot_style=None, fixed_cuts=None, plot_samples=None, cut_plot_variables=None, region=None,
    training_sampling=None, n_train_signal=None, n_train_background=None,
    bound_quantiles=(0.001, 0.999), preview_training=False,
):
    configure_style(plot_style)
    ensure_dir(outdir)
    var_list = list(variables.keys())
    directions = {v: variables[v]["direction"] for v in var_list}

    # Load data
    columns = list(dict.fromkeys(var_list + [weight_col]))
    if training_sampling is not None:
        for count in (n_train_signal, n_train_background):
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("training_sampling requires positive integer n_train_signal and n_train_background")
        if not 0 < train_ratio < 1:
            raise ValueError("Training sampling requires 0 < train_ratio < 1 for an internal TMVA test set")
        columns = list(dict.fromkeys(columns + ["sample", training_sampling["variable"]]))
    elif n_train_signal is not None or n_train_background is not None:
        raise ValueError("Exact n_train_signal/n_train_background require training_sampling")
    def load_training(path, size, seed):
        frame = pd.read_parquet(path, columns=columns)
        if training_sampling is None:
            return frame, None
        selected, report = sample_training_candidates(frame, size, training_sampling, seed, weight_col)
        return selected[var_list + [weight_col]], report

    if training_sampling is not None:
        n_signal = math.ceil(n_train_signal / train_ratio)
        n_background = math.ceil(n_train_background / train_ratio)
    sig_tr, signal_report = load_training(train_signal_path, n_signal, random_seed)
    bkg_tr, background_report = load_training(train_bkg_path, n_background, random_seed + 1)
    if training_sampling is not None:
        report = {"region": region, "tag": tag, "source": "prepared train partition",
                  "signal": dict(signal_report, path=os.path.abspath(train_signal_path),
                                 training_rows=n_train_signal, tmva_test_rows=n_signal - n_train_signal),
                  "background": dict(background_report, path=os.path.abspath(train_bkg_path),
                                     training_rows=n_train_background, tmva_test_rows=n_background - n_train_background)}
        with open(os.path.join(outdir, f"training_sampling_{tag}.json"), "w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        for role in ("signal", "background"):
            info = report[role]
            print(f"[TMVA] {region}/{role}: {info['training_rows']:,} training + "
                  f"{info['tmva_test_rows']:,} test from {info['available']:,} positive-weight prepared rows")
    for df in (sig_tr, bkg_tr):
        if not np.isfinite(df.to_numpy()).all() or (df[weight_col] < 0).any() or df[weight_col].sum() <= 0:
            raise ValueError("TMVA inputs must be finite with nonnegative training weights")

    # Initial bounds
    quantiles = np.asarray(bound_quantiles, dtype=float)
    if quantiles.shape != (2,) or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("bound_quantiles must contain two increasing probabilities in [0, 1]")
    bounds = {}
    bound_report = dict(region=region, tag=tag, quantiles=quantiles.tolist(),
                        quantile_weighting="unweighted", source="signal pool before the internal TMVA split",
                        signal_rows=len(sig_tr), background_rows=len(bkg_tr), variables={})
    for v in var_list:
        qlow, qhigh, q99, q999 = np.quantile(sig_tr[v], [*quantiles, .99, .999])
        b99, b999 = np.quantile(bkg_tr[v], [.99, .999])
        lo = max(variables[v].get("min", float(sig_tr[v].min())), qlow)
        hi = min(variables[v].get("max", float(sig_tr[v].max())), qhigh)
        if not np.isfinite([lo, hi]).all() or lo > hi:
            raise ValueError(f"Invalid initial bounds for {v}: {lo}, {hi}")
        bounds[v] = (float(lo), float(hi))
        bound_report["variables"][v] = dict(signal_q99=float(q99), signal_q999=float(q999),
                                            background_q99=float(b99), background_q999=float(b999),
                                            signal_max=float(sig_tr[v].max()), background_max=float(bkg_tr[v].max()),
                                            lower=float(lo), upper=float(hi))
    bounds_path = os.path.join(outdir, f"training_bounds_{tag}.json")
    with open(bounds_path, "w") as handle:
        json.dump(bound_report, handle, indent=2, allow_nan=False)
    print(f"[TMVA] {region}/{tag}: unweighted signal quantiles {quantiles.tolist()} set initial bounds")
    print(pd.DataFrame(bound_report["variables"]).T[
        ["signal_q99", "signal_q999", "background_q99", "lower", "upper"]].to_string(float_format=lambda x: f"{x:.6g}"))
    if preview_training:
        return {"bounds": bounds, "bounds_report": bounds_path}

    # Validation retains signed physical weights, independent of training reweighting.
    sig_va = pd.read_parquet(valid_signal_path)
    bkg_va = pd.read_parquet(valid_bkg_path)
    fixed_cuts = {name: normalize_fixed_cuts(cuts) for name, cuts in (fixed_cuts or {}).items()}
    # Check fixed-cut inputs before starting expensive optimization.
    for cuts in fixed_cuts.values():
        pass_fixed_cuts(sig_va, cuts)
        pass_fixed_cuts(bkg_va, cuts)

    WPs, roc_points = {}, []

    for idx, eff_target in enumerate(sorted(target_efficiencies, reverse=True), start=1):
        method_name = f"CutsGA_{tag}_wp{int(eff_target*100)}"
        weights_xml, window_cuts, selected_efficiency = tmva_train_one_wp_inmem(
            sig_df=sig_tr, bkg_df=bkg_tr,
            var_list=var_list, bounds=bounds,
            out_dir=outdir, method_name=method_name,
            ncycles=tmva_cycles, popsize=tmva_popsize,
            random_seed=random_seed,train_ratio=train_ratio,
            weight_col=weight_col, eff_target=eff_target,
            n_workers = n_workers,
            n_signal = n_signal,
            n_background = n_background,
            efficiency_tolerance=efficiency_tolerance,
            n_train_signal=n_train_signal,
            n_train_background=n_train_background,
        )

#        wps = parse_tmva_wps(weights_xml)
#        picked = pick_nearest_eff(wps, eff_target) if wps else None
#        if picked is None:
#            picked_eff = float("nan")
#            window_cuts = {v: (-np.inf, np.inf) for v in var_list}
#        else:
#            picked_eff = picked["eff"]
#            window_cuts = picked["window_cuts"]

        print(window_cuts)

        cuts_one_sided = to_one_sided(window_cuts, directions)
        wp_name = f"WP{int(eff_target*100)}"
        fixed = fixed_cuts.get(wp_name, {})
        ms = passes(sig_va, var_list, cuts_one_sided, directions) & pass_fixed_cuts(sig_va, fixed)
        mb = passes(bkg_va, var_list, cuts_one_sided, directions) & pass_fixed_cuts(bkg_va, fixed)
        eff_s_val = weighted_efficiency(ms, sig_va["weight"].values)
        eff_b_val = weighted_efficiency(mb, bkg_va["weight"].values)
        roc_points.append((eff_b_val, eff_s_val, eff_target))

        WPs[wp_name] = {
            "fixed_cuts": fixed,
            "eff_selected_from_tmva": selected_efficiency,
            "requested_efficiency": eff_target,
            "cuts": {v: None if not np.isfinite(cuts_one_sided[v]) else float(cuts_one_sided[v])
                     for v in var_list},
            "window_cuts": {v: ([-1e30, float(cuts_one_sided[v])] if directions[v] == "less"
                                else [float(cuts_one_sided[v]), 1e30]) for v in var_list},
            "strict": True,
            "isolation_denominator": isolation_denominator,
            "eff_val": eff_s_val,
            "rej_val": 1 - eff_b_val,
            "tmva_weights_xml": weights_xml,
            "bounds_used": {v: [float(bounds[v][0]), float(bounds[v][1])] for v in var_list},
        }

        # tighten bounds
        for v in var_list:
            thr = cuts_one_sided.get(v, np.nan)
            lo, hi = bounds[v]
            if np.isfinite(thr):
                if directions[v] == "less":
                    hi = min(hi, thr)
                else:
                    lo = max(lo, thr)
            bounds[v] = (lo, hi)

    wp_path = os.path.join(outdir, f"CutBased_WPs_{tag}_TMVA_seq.json")
    with open(wp_path, "w") as handle:
        json.dump(WPs, handle, indent=2)
    sources = {tag: load_configured_wps(wp_path)}
    plot_samples = plot_samples or {"signal": "Signal", "background": "Background"}
    region = region or os.path.basename(outdir)
    for variable, specification in dict(variables, **(cut_plot_variables or {})).items():
        samples = {plot_samples[role]: (frame[variable].to_numpy(), frame["weight"].to_numpy())
                   for role, frame in (("signal", sig_va), ("background", bkg_va))}
        fig, ax, report = plot_cut_distribution(samples, variable, specification, sources, region, plot_style)
        ax.text(.02, .98, region, transform=ax.transAxes, va="top", fontweight="bold", fontsize=18)
        path = os.path.join(outdir, f"{tag}_{variable}_WPs")
        with open(path + ".json", "w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        save_plot(fig, path + ".png", plot_style)

    roc_points = sorted(roc_points, key=lambda x: x[0])
    fig, ax = plt.subplots()
    plt.plot([p[0] for p in roc_points], [p[1] for p in roc_points], "o", linestyle="none", label="TMVA CutsGA (validation)")
    for fpr,tpr,eff in roc_points:
        plt.text(fpr, tpr, f"WP{int(eff*100)}", fontsize="x-small")
    plt.xlabel("Background efficiency (FPR)")
    plt.ylabel("Signal efficiency (TPR)")
    add_legend(ax, plot_style)
    save_plot(fig, os.path.join(outdir, f"{tag}_ROC_curve.png"), plot_style)

    print_id_table({region: sources},
                   os.path.join(outdir, f"CutBased_WPs_{tag}_TMVA_seq.md"))

    return {"WPs": WPs, "roc": os.path.join(outdir, f"{tag}_ROC_curve.png")}

# ---------------- API entry ----------------
def TrainCutBasedID_TMVAAnalysis(**kwargs):
    regions  = kwargs.pop("regions", [])
    inputdir = kwargs.pop("inputdir")
    variables_set = kwargs.pop("variables")
    outdir   = kwargs.pop("outputdir")
    target_efficiencies = kwargs.pop("target_efficiencies", [0.90, 0.80, 0.70])
    tmva_cycles = kwargs.pop("tmva_cycles", 60)
    tmva_popsize = kwargs.pop("tmva_popsize", 120)
    random_seed  = kwargs.pop("random_seed", 12345)
    train_ratio  = kwargs.pop("train_ratio", 0.8)
    n_workers    = kwargs.pop("n_workers", 1)
    n_signal = kwargs.pop("n_signal", 9e19)
    n_background = kwargs.pop("n_background", 9e19)
    weight_col = kwargs.pop("weight_col", "training_weight")
    for region in regions:
        region_out = os.path.join(outdir, region); 
        ensure_dir(region_out)
        print(f"[TMVA] region = {region}")
        for tag, variables in variables_set.items():
            train_cutbased_tmva_sequential_inmem(
                train_signal_path=os.path.join(inputdir, region, "train", "signal.parquet"),
                train_bkg_path=os.path.join(inputdir, region, "train", "background.parquet"),
                valid_signal_path=os.path.join(inputdir, region, "valid", "signal.parquet"),
                valid_bkg_path=os.path.join(inputdir, region, "valid", "background.parquet"),
                variables=variables,
                target_efficiencies=target_efficiencies,
                outdir=region_out,
                tmva_cycles=tmva_cycles,
                tmva_popsize=tmva_popsize,
                random_seed=random_seed,
                train_ratio=train_ratio,
                tag=tag,
                weight_col=weight_col,
                isolation_denominator=kwargs.get("isolation_denominator", "unspecified"),
                efficiency_tolerance=kwargs.get("efficiency_tolerance", 0.03),
                plot_style=kwargs.get("plot_style"),
                fixed_cuts=kwargs.get("fixed_cuts", {}).get(tag, {}),
                plot_samples=kwargs.get("plot_samples"),
                cut_plot_variables=kwargs.get("cut_plot_variables"),
                region=region,
                n_workers=n_workers,
                n_signal=n_signal,
                n_background=n_background,
                training_sampling=kwargs.get("training_sampling"),
                n_train_signal=kwargs.get("n_train_signal"),
                n_train_background=kwargs.get("n_train_background"),
                bound_quantiles=kwargs.get("bound_quantiles", (0.001, 0.999)),
                preview_training=kwargs.get("preview_training", False),
            )
    print("[TMVA] All regions finished.")
