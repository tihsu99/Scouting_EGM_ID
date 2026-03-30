import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed
import mplhep as hep
from correctionlib.schemav2 import Correction, CorrectionSet
import correctionlib.schemav2 as cs
import rich
plt.style.use(hep.style.CMS)

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
def fit_and_plot(args):
    (eta_min, eta_max, x_sel, y_sel, x_expr, y_expr,
     plotdir, fitted_func_str, var_name, y_type, x_bins,
     x_title, y_title, title) = args

    if len(x_sel) == 0:
        print(f"⚠️ No entries in η ∈ [{eta_min}, {eta_max})")
        return None

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
            if np.any(mask):
                if y_type.lower() == "mean":
                    y_fit_data[i] = np.mean(y_sel[mask])
                elif "contour" in y_type.lower():
                    quantile_level = int(y_type.lower().replace("contour", ""))/100
                    y_fit_data[i] = np.quantile(y_sel[mask], quantile_level)  # median, can add ± quantiles for error
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
    p0 = np.ones(len(params))
    popt, pcov = curve_fit(func, x_fit_data, y_fit_data, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    # x_fit_data and y_fit_data are your prepared arrays
    p0 = [1.0, 0.0]  # initial guesses for A and b
    popt, pcov = curve_fit(linear_func, x_fit_data, y_fit_data, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    A, b = popt
    A_err, b_err = perr



    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    hep.cms.label("Preliminary", ax=ax, loc=0)
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
    ax.legend()

    # Annotate fit
    txt = [f"range: {eta_min} < $|\eta|$ < {eta_max}", f"Fit: y = {expr}"]
    for n, v, e in zip(params, popt, perr):
        txt.append(f"{n} = {v:.4g} ± {e:.4g}")
    ax.text(0.75, 0.25, "\n".join(txt), transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    os.makedirs(os.path.join(plotdir, f"{var_name}"), exist_ok=True)
    outfile = os.path.join(plotdir, f"{var_name}", f"EffArea_eta_{eta_min}_{eta_max}.png")
    fig.tight_layout()
    fig.savefig(outfile)
    plt.close(fig)
    print(f"✅ Saved plot: {outfile}")
    return {"value": A}


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

    # --- Build file list ---
    fileset = []
    for category, cat_info in process.items():
        if not(category == "signal"):
            continue
        samples = cat_info["samples"]
        for sample_name, _ in samples.items():
            sample_path = os.path.join(inputdir, sample_name)
            files = [
                os.path.join(sample_path, f)
                for f in os.listdir(sample_path)
                if f.endswith(".parquet")
            ]
            if n_files > 0:
                files = files[:min(n_files, len(files))]
            fileset.extend(files)

    print(f"📦 Found {len(fileset)} parquet files")

    # --- Load data in parallel ---
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        dfs = list(ex.map(load_parquet_file, fileset))

    df_all = pd.concat(dfs, ignore_index=True)
    df_all =  df_all.query(requirement)
    print(f"✅ Loaded {len(df_all):,} rows")

    # --- Config extraction ---
    corrections = []

    for var_name, inspect_dict in inspect_variable.items():
        x_expr = inspect_dict["x"]
        y_expr = inspect_dict["y"]
        eta_bins = inspect_dict["eta_bins"]
        fitted_func_str = inspect_dict["fitted_func"]
        y_type = inspect_dict["y_type"]
        x_bins = inspect_dict["x_bins"]
        x_title = inspect_dict["x_title"]
        y_title = inspect_dict["y_title"]
        title = inspect_dict["title"]
        df_all[var_name] = 0

        if len(x_bins) == 3:
            x_bins = np.linspace(x_bins[0], x_bins[1], x_bins[2]+1)

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
            mask = (abs(df_all["eta"]) >= eta_min) & (abs(df_all["eta"]) < eta_max)
            x_sel = df_all.loc[mask, f"{var_name}_xval"].to_numpy()
            y_sel = df_all.loc[mask, f"{var_name}_yval"].to_numpy()
            tasks.append((eta_min, eta_max, x_sel, y_sel, x_expr, y_expr, plotdir, fitted_func_str, var_name, y_type, x_bins, x_title, y_title, title))

        # --- Run parallel fits ---
        results = []
        with ProcessPoolExecutor(max_workers=min(n_workers, len(tasks))) as ex:
            futures = [ex.submit(fit_and_plot, t) for t in tasks]
            for fut, task in zip(as_completed(futures), tasks):
                res = fut.result()
                eta_min, eta_max = task[0], task[1]
                if res:
                    # Optionally attach eta info to the result dict
                    res["eta_min"] = eta_min
                    res["eta_max"] = eta_max
                    results.append(res)

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


