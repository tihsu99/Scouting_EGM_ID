#!/usr/bin/env python3
import os, json, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET

import ROOT
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
    """Return list of {eff, window_cuts: {var:(min,max)}} read from TMVA weights."""
    res = []
    try:
        tree = ET.parse(weights_xml_path)
        root = tree.getroot()
    except Exception as e:
        warnings.warn(f"Failed to read TMVA XML {weights_xml_path}: {e}")
        return res

    # Most common layout: nodes with attribute eff and child Var(Name/Expression, Min/Max)
    for block in root.findall(".//*[@eff]"):
        eff_attr = block.get("eff")
        try:
            eff = float(eff_attr)
        except Exception:
            continue
        cuts = {}
        for vn in block.findall(".//Var"):
            name = vn.get("Name") or vn.get("Expression") or vn.get("Label")
            vmin = vn.get("Min"); vmax = vn.get("Max")
            if name is None: continue
            vmin = float(vmin) if vmin is not None else -np.inf
            vmax = float(vmax) if vmax is not None else np.inf
            cuts[name] = (vmin, vmax)
        if cuts: res.append({"eff": eff, "window_cuts": cuts})

    # sort & de-dup
    res.sort(key=lambda d: d["eff"])
    out, seen = [], set()
    for r in res:
        k = round(r["eff"], 5)
        if k not in seen:
            seen.add(k); out.append(r)
    return out

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
                            train_ratio, method_name, ncycles, popsize, random_seed, eff_target, n_signal, n_background, n_workers=1):
    """
    sig_df, bkg_df: pandas DataFrames
    bounds: {var: (lo, hi)} -> will be passed to CutsGA as CutRangeMin/Max
    Returns path to weights xml.
    """
    ensure_dir(out_dir)
    #ROOT.EnableImplicitMT()  # multi-CPU (OpenMP inside TMVA)
    ROOT.TMVA.gConfig().fOMPThreads = n_workers

    outfile_path = os.path.join(out_dir, f"TMVA_{method_name}.root")
    output = ROOT.TFile(outfile_path, "RECREATE")

    factory = TMVA.Factory(
        "TMVAClassification",
        output,
        "!V:!Silent:Color:DrawProgressBar:Transformations=I"
    )

    dl = TMVA.DataLoader("dataset")

    var_list_extend = [v for v in var_list]
    if weight_col is not None:
        var_list_extend.append(weight_col)
    for v in var_list:
        dl.AddVariable(v, "F")

    # Convert pandas→RDataFrame→TTree (in memory, no .root saved)
    sig_rdf = ROOT.RDF.MakeNumpyDataFrame({c: sig_df[c].to_numpy() for c in var_list_extend})
    bkg_rdf = ROOT.RDF.MakeNumpyDataFrame({c: bkg_df[c].to_numpy() for c in var_list_extend})
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
    n_train_sig = int(n_sig_total * train_ratio)
    n_train_bkg = int(n_bkg_total * train_ratio)
    n_test_sig  = n_sig_total - n_train_sig
    n_test_bkg  = n_bkg_total - n_train_bkg


    dl.PrepareTrainingAndTestTree(TCut(""), TCut(""),
        f"nTrain_Signal={n_train_sig}:nTrain_Background={n_train_bkg}:nTest_Signal={n_test_sig}:nTest_Background={n_test_bkg}:SplitMode=Random:!V")

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

    output.Close()
    # Optional cleanup
    sigFile.Close()
    bkgFile.Close()
    os.remove(sig_path)
    os.remove(bkg_path)
    weights_xml = os.path.join(out_dir, "dataset", "weights",
                               f"TMVAClassification_{method_name}.weights.xml")
    if not os.path.exists(weights_xml):
        alt = os.path.join("dataset", "weights", f"TMVAClassification_{method_name}.weights.xml")
        if os.path.exists(alt): weights_xml = alt


    method = factory.GetMethod("dataset", method_name)
    # Explicitly cast to MethodCuts
    method_cuts = TMVA.MethodCuts.Class().DynamicCast(TMVA.MethodCuts.Class(), method)
    #if not method_cuts:
    #    raise RuntimeError("Failed to cast method to TMVA::MethodCuts")
    nvars = len(var_list)   # number of input variables

    cutLo = ROOT.std.vector('double')(nvars)
    cutHi = ROOT.std.vector('double')(nvars)
    method_cuts.GetCuts(float(eff_target), cutLo, cutHi)

    output_cut = dict()
    for i, v in enumerate(var_list):
        lo, hi = bounds[v]
        output_cut[v] = (cutLo[i], cutHi[i])
    return weights_xml, output_cut

# ---------------- Full sequential-bound pipeline ----------------
def train_cutbased_tmva_sequential_inmem(
    train_signal_path, train_bkg_path, valid_signal_path, valid_bkg_path,
    variables, target_efficiencies, outdir, train_ratio=0.8, weight_col="weight",
    tmva_cycles=60, tmva_popsize=120, random_seed=12345, tag="Default", n_workers = 1, n_signal = 9e19, n_background=9e19
):
    ensure_dir(outdir)
    var_list = list(variables.keys())
    directions = {v: variables[v]["direction"] for v in var_list}

    # Load data
    def load_and_keep(path):
        df = pd.read_parquet(path)
        keep = {v: df[v].values for v in var_list}
        keep[weight_col] = df[weight_col].values if weight_col in df.columns else np.ones(len(df))
        return pd.DataFrame(keep)

    sig_tr = load_and_keep(train_signal_path)
    bkg_tr = load_and_keep(train_bkg_path)
    sig_va = load_and_keep(valid_signal_path)
    bkg_va = load_and_keep(valid_bkg_path)

    # Initial bounds
    bounds = {}
    for v in var_list:
        lo = max(variables[v].get("min", float(np.nanmin(sig_tr[v]))), np.quantile(sig_tr[v], 0.001))
        hi = min(variables[v].get("max", float(np.nanmax(sig_tr[v]))), np.quantile(sig_tr[v], 0.999))
        bounds[v] = (float(lo), float(hi))

    WPs, roc_points = {}, []

    for idx, eff_target in enumerate(sorted(target_efficiencies, reverse=True), start=1):
        method_name = f"CutsGA_wp{int(eff_target*100)}"
        weights_xml, window_cuts = tmva_train_one_wp_inmem(
            sig_df=sig_tr, bkg_df=bkg_tr,
            var_list=var_list, bounds=bounds,
            out_dir=outdir, method_name=method_name,
            ncycles=tmva_cycles, popsize=tmva_popsize,
            random_seed=random_seed,train_ratio=train_ratio,
            weight_col=weight_col, eff_target=eff_target,
            n_workers = n_workers,
            n_signal = n_signal,
            n_background = n_background
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
        ms = passes(sig_va, var_list, cuts_one_sided, directions)
        mb = passes(bkg_va, var_list, cuts_one_sided, directions)
        eff_s_val = weighted_efficiency(ms, sig_va[weight_col].values)
        eff_b_val = weighted_efficiency(mb, bkg_va[weight_col].values)
        roc_points.append((eff_b_val, eff_s_val, eff_target))

        WPs[f"WP{int(eff_target*100)}"] = {
            "eff_selected_from_tmva": eff_target,
            "cuts": {v: None if not np.isfinite(cuts_one_sided[v]) else float(cuts_one_sided[v])
                     for v in var_list},
            "window_cuts": {v: [float(window_cuts.get(v, (-np.inf, np.inf))[0]),
                                float(window_cuts.get(v, (-np.inf, np.inf))[1])]
                            for v in var_list},
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

    # --- Plots ---
    for v, vconf in variables.items():
        bins = vconf["binning"]
        if isinstance(bins, (list, tuple)) and len(bins) == 3:
            vmin, vmax, nbins = bins
            bins = np.linspace(vmin, vmax, nbins + 1)
        direction = vconf.get("direction", "less")

        plt.figure(figsize=(10, 8))
        sns.histplot(sig_va, x=v, weights=sig_va[weight_col], label="Signal",
                     stat="density", bins=bins, kde=False, alpha=0.5, color="C0")
        sns.histplot(bkg_va, x=v, weights=bkg_va[weight_col], label="Background",
                     stat="density", bins=bins, kde=False, alpha=0.5, color="C1")
        palette = ["#0072B2","#D55E00","#009E73","#CC79A7","#F0E442","#56B4E9","#E69F00","#999999"]

        for i,(wp, info) in enumerate(WPs.items()):
            thr = info["cuts"].get(v, None)
            if thr is None: continue
            plt.axvline(thr, linestyle="--", linewidth=2, color=palette[i % len(palette)],
                        label=f"{wp} ({thr:.3f})")
            y1 = plt.ylim()[1]*0.85
            vmin, vmax = np.min(sig_va[v]), np.max(sig_va[v])
            if direction == "less":
                plt.annotate("", xy=(thr-0.02*(vmax-vmin), y1),
                             xytext=(thr-0.12*(vmax-vmin), y1),
                             arrowprops=dict(arrowstyle="->", lw=1.5, color="k"))
            else:
                plt.annotate("", xy=(thr+0.02*(vmax-vmin), y1),
                             xytext=(thr+0.12*(vmax-vmin), y1),
                             arrowprops=dict(arrowstyle="->", lw=1.5, color="k"))

        plt.title(f"{v} with sequential WPs (TMVA CutsGA)")
        plt.xlabel(v); plt.ylabel("Density"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{tag}_{v}_WPs.png"))
        plt.close()

    roc_points = sorted(roc_points, key=lambda x: x[0])
    plt.figure(figsize=(8, 8))
    plt.plot([p[0] for p in roc_points], [p[1] for p in roc_points], "o-", label="TMVA CutsGA (validation)")
    for fpr,tpr,eff in roc_points:
        plt.text(fpr, tpr, f"WP{int(eff*100)}", fontsize=8)
    plt.xlabel("Background efficiency (FPR)")
    plt.ylabel("Signal efficiency (TPR)")
    plt.title("Cut-based ID ROC (Validation) — TMVA sequential bounds")
    plt.grid(True, ls="--", alpha=0.4); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{tag}_ROC_curve.png")); plt.close()

    with open(os.path.join(outdir, f"CutBased_WPs_{tag}_TMVA_seq.json"), "w") as f:
        json.dump(WPs, f, indent=2)

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
                n_workers=n_workers,
                n_signal=n_signal,
                n_background=n_background
            )
    print("[TMVA] All regions finished.")

