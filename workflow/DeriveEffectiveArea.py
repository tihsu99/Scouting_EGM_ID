import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from correctionlib.schemav2 import Correction, CorrectionSet
import correctionlib.schemav2 as cs
import rich
from workflow.data import edges, event_split, load_samples, training_weights
from workflow.plotting import add_legend, configure_style, save_plot

# === Dynamic function creation ===
def parse_fitted_function(fitted_func_str):
    """
    Convert 'y = A1*x + A2' into a Python callable and list of parameter names.
    """
    match = re.search(r"y\s*=\s*(.+)", fitted_func_str)
    if not match:
        raise ValueError(f"Invalid fitted_func format: {fitted_func_str}")

    expr = match.group(1).strip()

    # Extract parameter names (A1, A2, ...)
    params = sorted(set(re.findall(r"\bA\d+\b", expr)), key=lambda s: int(s[1:]))
    param_str = ", ".join(params)

    # Define function dynamically
    func_code = f"lambda x, {param_str}: {expr}"
    func = eval(func_code, {"np": np, "__builtins__": {}})

    return func, params, expr

def linear_func(x, A, b):
    return A * x + b

# === Parallel fit and plot ===
def fit_and_plot(args, plot_style=None):
    (eta_min, eta_max, x_sel, y_sel, x_expr, y_expr,
     plotdir, fitted_func_str, var_name, y_type, x_bins,
     x_title, y_title, title, weights, coefficient) = args
    configure_style(plot_style)

    if len(x_sel) == 0:
        print(f"⚠️ No entries in η ∈ [{eta_min}, {eta_max})")
        raise ValueError(f"No calibration entries in eta [{eta_min}, {eta_max})")

    # --- Prepare data depending on type ---
    if y_type.lower() == "none":
        # scatter
        x_fit_data = x_sel
        y_fit_data = y_sel
    else:
        # bin x for profiling
        bins = x_bins
        centers = 0.5 * (bins[1:] + bins[:-1])
        y_fit_data = np.zeros_like(centers)
        for i in range(len(bins)-1):
            mask = (x_sel >= bins[i]) & (x_sel < bins[i+1])
            if np.any(mask) and weights[mask].sum() > 0:
                if y_type.lower() == "mean":
                    y_fit_data[i] = np.average(y_sel[mask], weights=weights[mask])
                elif "contour" in y_type.lower():
                    quantile_level = int(y_type.lower().replace("contour", ""))/100
                    order = np.argsort(y_sel[mask])
                    values, w = y_sel[mask][order], weights[mask][order]
                    positions = (np.cumsum(w) - 0.5 * w) / w.sum()
                    y_fit_data[i] = np.interp(quantile_level, positions, values)
                else:
                    raise ValueError(f"Unknown plot type: {y_type}")
            else:
                y_fit_data[i] = np.nan
        # remove bins with nan
        mask_valid = ~np.isnan(y_fit_data)
        x_fit_data = centers[mask_valid]
        y_fit_data = y_fit_data[mask_valid]

    # --- Fit function ---
    func, params, expr = parse_fitted_function(fitted_func_str)
    if coefficient not in params or len(x_fit_data) <= len(params):
        raise ValueError(f"{var_name}, eta [{eta_min}, {eta_max}): insufficient fit bins or unknown coefficient")
    popt, pcov = curve_fit(func, x_fit_data, y_fit_data, p0=np.ones(len(params)), maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    if not np.isfinite(popt).all() or not np.isfinite(perr).all():
        raise ValueError(f"Invalid fit for {var_name} in eta [{eta_min}, {eta_max})")

    # --- Plot ---
    fig, ax = plt.subplots()
    if y_type.lower() == "none":
        ax.scatter(x_sel, y_sel, s=5, alpha=0.4, label="Data")
    elif y_type.lower() == "mean":
        ax.errorbar(x_fit_data, y_fit_data, fmt="o", label="Profile mean")
    elif "contour" in y_type.lower():
        quantile_level = int(y_type.lower().replace("contour", ""))
        ax.plot(x_fit_data, y_fit_data, "o", label=f"Quantile {quantile_level}%")
    else:
        raise ValueError(f"Unknown plot type: {y_type}")

    # Fit line
    x_fit = np.linspace(np.nanmin(x_fit_data), np.nanmax(x_fit_data), 300)
    ax.plot(x_fit, func(x_fit, *popt), color="red", label="Fit")

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    add_legend(ax, plot_style)

    # Annotate fit
    txt = [rf"range: {eta_min} < $|\eta|$ < {eta_max}", f"Fit: y = {expr}"]
    for n, v, e in zip(params, popt, perr):
        txt.append(f"{n} = {v:.4g} ± {e:.4g}")
    ax.text(0.95, 0.05, "\n".join(txt), transform=ax.transAxes,
            va="bottom", ha="right", fontsize="small",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    os.makedirs(os.path.join(plotdir, f"{var_name}"), exist_ok=True)
    outfile = os.path.join(plotdir, f"{var_name}", f"EffArea_eta_{eta_min}_{eta_max}.png")
    save_plot(fig, outfile, plot_style)
    print(f"✅ Saved plot: {outfile}")
    return {"value": float(popt[params.index(coefficient)]), "eta_min": eta_min, "eta_max": eta_max,
            "parameters": dict(zip(params, popt.tolist())), "errors": dict(zip(params, perr.tolist())),
            "fit_bins": len(x_fit_data), "entries": len(x_sel)}


# === Parallel parquet loading ===
def load_parquet_file(f):
    return pd.read_parquet(f)


# === Main analysis ===
def DeriveEffectiveAreaAnalysis(**kwargs):
    """
    Flexible multi-CPU framework for deriving effective area corrections.
    Accepts any function form in 'fitted_func' (e.g. y = A1*x + A2*x**2).
    """

    inputdir = kwargs.pop("inputdir")
    process = kwargs.pop("process")
    plotdir = kwargs.pop("plotdir")
    inspect_variable = kwargs.pop("EffectiveArea")
    n_workers = kwargs.pop("n_workers", 8)
    n_files = kwargs.pop("n_files", -1)
    requirement = kwargs.pop("requirement", None)


    os.makedirs(plotdir, exist_ok=True)
    print(f"✅ [DeriveEffectiveArea] Plot dir: {plotdir}")

    limit = kwargs.get("max_rows_per_sample")
    if limit is not None and limit != -1:
        print(f"Keeping at most {limit:,} input candidates per sample before cuts and splitting")
    frames, report = load_samples(
        inputdir, process, n_files, kwargs.get("normalization"),
        columns=kwargs.get("input_columns"), max_rows_per_sample=limit,
        batch_size=kwargs.get("batch_size", 100000),
        sampling=kwargs.get("sampling"), progress=kwargs.get("progress", False),
    )
    for sample, info in report.items():
        print(f"Read {info['rows']:,} input candidates from {sample}")
    df_all = pd.concat(frames.values(), ignore_index=True)
    del frames
    if requirement:
        df_all = df_all.query(requirement).copy()
    if kwargs.get("calibration_train_only", False):
        df_all, _ = event_split(df_all, kwargs.get("train_ratio", 0.8), kwargs.get("random_seed", 42),
                               kwargs.get("split_columns"))
    df_all["calibration_weight"] = training_weights(df_all, kwargs.get("calibration_weight_mode", "nonnegative"))
    print(f"Loaded {len(df_all):,} calibration rows")
    for sample, info in report.items():
        weights = df_all.loc[df_all["sample"] == sample, "calibration_weight"].to_numpy()
        info.update(calibration_rows=len(weights), calibration_sumw=float(weights.sum()),
                    calibration_effective_entries=float(weights.sum() ** 2 / np.square(weights).sum())
                    if np.square(weights).sum() > 0 else 0.0)
    with open(os.path.join(plotdir, "normalization.json"), "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    # --- Config extraction ---
    corrections = []

    for var_name, inspect_dict in inspect_variable.items():
        x_expr = inspect_dict["x"]
        y_expr = inspect_dict["y"]
        eta_bins = inspect_dict["eta_bins"]
        fitted_func_str = inspect_dict["fitted_func"]
        y_type = inspect_dict["y_type"]
        x_bins_by_eta = inspect_dict.get("x_bins_by_eta")
        if x_bins_by_eta is None:
            x_bins_by_eta = [None] * (len(eta_bins) - 1)
        if not isinstance(x_bins_by_eta, (list, tuple)) or len(x_bins_by_eta) != len(eta_bins) - 1:
            raise ValueError(f"{var_name}: x_bins_by_eta must have {len(eta_bins) - 1} entries, "
                             "one per eta interval; use null to inherit x_bins")
        x_bins_by_eta = [edges(inspect_dict["x_bins"] if spec is None else spec)
                         for spec in x_bins_by_eta]
        x_title = inspect_dict["x_title"]
        y_title = inspect_dict["y_title"]
        title = inspect_dict["title"]
        df_all[var_name] = 0.0

        print(f"📊 Fitting {y_expr} vs {x_expr}")
        print(f"η bins: {eta_bins}")
        print(f"Function: {fitted_func_str}")

        # --- Evaluate expressions ---
        df_all[f"{var_name}_xval"] = df_all.eval(x_expr)
        df_all[f"{var_name}_yval"] = df_all.eval(y_expr)
        if "eta" not in df_all.columns:
            raise KeyError("Column 'eta' not found in data!")

        # --- Build tasks per η-bin ---
        tasks = []
        for i in range(len(eta_bins) - 1):
            eta_min, eta_max = eta_bins[i], eta_bins[i + 1]
            x_bins = x_bins_by_eta[i]
            mask = (abs(df_all["eta"]) >= eta_min) & (abs(df_all["eta"]) < eta_max)
            x_sel = df_all.loc[mask, f"{var_name}_xval"].to_numpy()
            y_sel = df_all.loc[mask, f"{var_name}_yval"].to_numpy()
            tasks.append((eta_min, eta_max, x_sel, y_sel, x_expr, y_expr, plotdir, fitted_func_str, var_name, y_type, x_bins, x_title, y_title, title,
                          df_all.loc[mask, "calibration_weight"].to_numpy(), inspect_dict.get("coefficient", "A1")))

        # --- Run parallel fits ---
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as ex:
            # map preserves eta-bin order even when worker completion order differs.
            results = list(ex.map(fit_and_plot, tasks, repeat(kwargs.get("plot_style"))))

        # --- Fill df_all with fit results ---
        for res in results:
            eta_min, eta_max = res["eta_min"], res["eta_max"]
            mask = (abs(df_all["eta"]) >= eta_min) & (abs(df_all["eta"]) < eta_max)
            df_all.loc[mask, f"{var_name}"] = res["value"]

        eta_edges = [res["eta_min"] for res in results]
        eta_edges.append(results[-1]["eta_max"])
        values = [res["value"] for res in results]
        print(eta_edges, values)
        corr = Correction(
            name=var_name,
            version=1,
            inputs=[cs.Variable(name="eta", type="real", description="")],
            output=cs.Variable(name=var_name, type="real"),
            data=cs.Binning(
                nodetype="binning",
                input="eta",
                edges=eta_edges,
                content=values,
                 flow="clamp",
            ),
        )
        rich.print(corr)
        corrections.append(corr)


        # --- Summary ---
        summary_df = pd.DataFrame(results)
        summary_csv = os.path.join(plotdir, f"{var_name}_fit_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        print(f"📄 Saved summary: {summary_csv}")
    # wrap in CorrectionSet
    corrset = CorrectionSet(schema_version=2, corrections=corrections)

    # save to disk
    with open(os.path.join(plotdir,"corrections_all.json"), "w") as f:
        f.write(corrset.json())


