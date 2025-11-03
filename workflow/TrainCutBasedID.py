import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from deap import base, creator, tools, algorithms
import optuna
from rich.console import Console
from rich.table import Table
from rich.markup import escape

console = Console()


# ---------------------------
# Utilities
# ---------------------------
def weighted_efficiency(mask, weights):
    if len(mask) == 0:
        return 0.0
    return np.sum(weights[mask]) / np.sum(weights)


def passes(df, variables, cuts, dirs):
    conds = []
    for i, v in enumerate(variables):
        thr = cuts[i]
        conds.append(df[v] < thr if dirs[v] == "less" else df[v] > thr)
    if not conds:
        return np.ones(len(df), dtype=bool)
    return np.logical_and.reduce(conds)


# ---------------------------
# GA optimization
# ---------------------------
def ga_optimize(sig_df, bkg_df, variables, dirs, weights_sig, weights_bkg,
                target_eff, bounds, n_gen=50, pop_size=40,
                cxpb=0.6, mutpb=0.3):
    """Run GA optimization and return best cuts + history"""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    def init_ind():
        return [np.random.uniform(lo, hi) for (lo, hi) in bounds]
    toolbox.register("individual", tools.initIterate, creator.Individual, init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        cuts = ind
        mask_sig = passes(sig_df, variables, cuts, dirs)
        mask_bkg = passes(bkg_df, variables, cuts, dirs)
        eff_s = weighted_efficiency(mask_sig, weights_sig)
        eff_b = weighted_efficiency(mask_bkg, weights_bkg)
        fom = eff_s / np.sqrt(eff_b + 1e-9)
        return (fom,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("mean", np.mean)

    logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb,
                                  ngen=n_gen, stats=stats, halloffame=hof, verbose=False)
    # --- Evaluate all individuals and store their efficiencies ---
    results = []
    for ind in pop:
        cuts = ind
        mask_sig = passes(sig_df, variables, cuts, dirs)
        mask_bkg = passes(bkg_df, variables, cuts, dirs)
        eff_s = weighted_efficiency(mask_sig, weights_sig)
        eff_b = weighted_efficiency(mask_bkg, weights_bkg)
        fom = eff_s / np.sqrt(eff_b + 1e-9)
        results.append((cuts, eff_s, eff_b, fom))

    # --- Convert to DataFrame for easier analysis ---
    import pandas as pd
    df_res = pd.DataFrame(results, columns=["cuts", "eff_s", "eff_b", "fom"])

    # --- Select candidate closest to target_eff ---
    df_res["delta_eff"] = abs(df_res["eff_s"] - target_eff)
    df_res = df_res.sort_values(by=["delta_eff", "eff_b", "fom"], ascending=[True, True, False])
    best = df_res.iloc[0]

    best_cuts = best["cuts"]
    best_eff_s = best["eff_s"]
    best_eff_b = best["eff_b"]

    print(f"✅ Best candidate: eff_s={best_eff_s:.4f}, eff_b={best_eff_b:.4f}, cuts={best_cuts}")

    del creator.FitnessMax, creator.Individual
    return best_cuts, logbook


# ---------------------------
# Optuna wrapper
# ---------------------------
def objective(trial, sig_df, bkg_df, variables, dirs, weights_sig, weights_bkg,
              bounds, target_eff):
    pop_size = trial.suggest_int("pop_size", 30, 150)
    n_gen = trial.suggest_int("n_gen", 30, 100)
    cxpb = trial.suggest_float("cxpb", 0.4, 0.9)
    mutpb = trial.suggest_float("mutpb", 0.1, 0.6)

    cuts, _ = ga_optimize(sig_df, bkg_df, variables, dirs, weights_sig,
                                weights_bkg, target_eff, bounds,
                                n_gen=n_gen, pop_size=pop_size,
                                cxpb=cxpb, mutpb=mutpb)

    mask_sig = passes(sig_df, variables, cuts, dirs)
    mask_bkg = passes(bkg_df, variables, cuts, dirs)
    eff_s = weighted_efficiency(mask_sig, weights_sig)
    eff_b = weighted_efficiency(mask_bkg, weights_bkg)
    return 1-eff_b


def tune_ga_with_optuna(sig_df, bkg_df, variables, dirs, weights_sig, weights_bkg,
                        bounds, target_eff, n_trials=5):
    study = optuna.create_study(direction="maximize")
    func = partial(objective, sig_df=sig_df, bkg_df=bkg_df,
                   variables=variables, dirs=dirs,
                   weights_sig=weights_sig, weights_bkg=weights_bkg,
                   bounds=bounds,
                   target_eff=target_eff)
    study.optimize(func, n_trials=n_trials, n_jobs=1)
    return study.best_params, study


# ---------------------------
# Full Training Function
# ---------------------------
def train_cutbased_ga(train_signal_path, train_bkg_path,
                      valid_signal_path, valid_bkg_path,
                      variables, target_efficiencies,
                      outdir,
                      weight_col="weight", n_trials=5):
    os.makedirs(outdir, exist_ok=True)
    console.print(f"[cyan]Training Cut-based GA (Optuna-tuned) for {outdir}[/cyan]")

    cols = list(variables.keys()) + [weight_col]
    sig_tr = pd.read_parquet(train_signal_path, columns=cols)
    bkg_tr = pd.read_parquet(train_bkg_path, columns=cols)
    sig_va = pd.read_parquet(valid_signal_path, columns=cols)
    bkg_va = pd.read_parquet(valid_bkg_path, columns=cols)


    for df in [sig_tr, bkg_tr, sig_va, bkg_va]:
        df.reset_index(drop=True, inplace=True)

        if weight_col not in df:
            df[weight_col] = 1.0

    weights_sig = sig_tr[weight_col].values
    weights_bkg = bkg_tr[weight_col].values

    bounds = [(float(variables[v]["min"]), float(variables[v]["max"])) for v in variables]
    WPs, roc_points = {}, []
    prev_cuts = None

    variable_directions = {v: info["direction"] for v, info in variables.items()}

    for eff in sorted(target_efficiencies, reverse=True):
        console.print(f"[bold green]▶ Optimizing for eff={eff:.2f}[/bold green]")
        if prev_cuts is not None:
            for i, v in enumerate(variables):
                lo, hi = bounds[i]
                if variable_directions[v] == "less":
                    hi = min(hi, prev_cuts[i])
                else:
                    lo = max(lo, prev_cuts[i])
                bounds[i] = (lo, hi)
        else:
            for i, v in enumerate(variables):
                lo, hi = bounds[i]
                if variable_directions[v] == "less":
                    hi = min(hi, np.quantile(sig_tr[v], 0.999))
                else:
                    lo = max(lo, np.quantile(sig_tr[v], 0.999))
                bounds[i] = (lo, hi)


        best_hparams, study = tune_ga_with_optuna(sig_tr, bkg_tr, list(variables.keys()),
                                                  variable_directions, weights_sig, weights_bkg,
                                                  bounds, eff, n_trials=n_trials)
        console.print(f"[yellow]Best Optuna params:[/yellow] {best_hparams}")

        cuts, _ = ga_optimize(sig_tr, bkg_tr, list(variables.keys()), variable_directions,
                                    weights_sig, weights_bkg, eff, bounds,
                                    n_gen=best_hparams["n_gen"], pop_size=best_hparams["pop_size"],
                                    cxpb=best_hparams["cxpb"], mutpb=best_hparams["mutpb"])

        mask_sig_va = passes(sig_va, list(variables.keys()), cuts, variable_directions)
        mask_bkg_va = passes(bkg_va, list(variables.keys()), cuts, variable_directions)
        eff_s_val = weighted_efficiency(mask_sig_va, sig_va[weight_col].values)
        eff_b_val = weighted_efficiency(mask_bkg_va, bkg_va[weight_col].values)
        roc_points.append((eff_b_val, eff_s_val, eff))

        WPs[f"WP{int(eff*100)}"] = {
            "cuts": dict(zip(variables.keys(), cuts)),
            "eff_val": eff_s_val,
            "rej_val": 1 - eff_b_val,
            "optuna_best": best_hparams,
        }
        prev_cuts = cuts


        for v, vconf in variables.items():
             # binning and direction
             bins = vconf["binning"]
             if len(bins) == 3:
                 vmin, vmax, nbins = bins[0], bins[1], bins[2]
                 bins = np.linspace(vmin, vmax, nbins + 1)
             direction = vconf.get("direction", "less")

             plt.figure(figsize=(6, 4))

             sns.histplot(
                 data=sig_va, x=v, weights=sig_va[weight_col],
                 label="Signal", color="C0", stat="density", bins=bins, kde=False, alpha=0.5
             )
             sns.histplot(
                 data=bkg_va, x=v, weights=bkg_va[weight_col],
                 label="Background", color="C1", stat="density", bins=bins, kde=False, alpha=0.5
             )

            
             colorblind_palette = [
                 "#0072B2",  # Blue
                 "#D55E00",  # Vermillion
                 "#009E73",  # Bluish green
                 "#CC79A7",  # Reddish purple
                 "#F0E442",  # Yellow
                 "#56B4E9",  # Sky blue
                 "#E69F00",  # Orange
                 "#999999",  # Gray
             ]


             # Draw all WP cuts
             for wp_idx, (wp_name, wp_info) in enumerate(WPs.items()):
                cut_val = wp_info["cuts"][v]

                plt.axvline(cut_val, linestyle="--", linewidth=2,
                    label=f"{wp_name} ({cut_val:.3f})", color=colorblind_palette[wp_idx])

                # Arrow to indicate direction
                ylims = plt.ylim()
                arrow_y = ylims[1] * 0.85
                if direction == "less":
                    plt.annotate("", xy=(cut_val - 0.02 * (vmax - vmin), arrow_y),
                         xytext=(cut_val - 0.12 * (vmax - vmin), arrow_y),
                         arrowprops=dict(arrowstyle="->", lw=1.5, color="k"))
                else:  # greater
                     plt.annotate("", xy=(cut_val + 0.02 * (vmax - vmin), arrow_y),
                         xytext=(cut_val + 0.12 * (vmax - vmin), arrow_y),
                         arrowprops=dict(arrowstyle="->", lw=1.5, color="k"))

             plt.title(f"{v} distribution with WPs")
             plt.xlabel(v)
             plt.ylabel("Density")
             plt.legend()
             plt.tight_layout()
             plt.savefig(os.path.join(outdir, f"{v}_WPs.png"), dpi=150)
             plt.close()

    # ROC
    roc_points = sorted(roc_points, key=lambda x: x[0])
    plt.figure(figsize=(6, 5))
    plt.plot([p[0] for p in roc_points], [p[1] for p in roc_points], "o-", label="GA Cutbased (Optuna-tuned)")
    for fpr, tpr, eff in roc_points:
        plt.text(fpr, tpr, f"WP{int(eff*100)}", fontsize=8)
    plt.xlabel("Background efficiency (FPR)")
    plt.ylabel("Signal efficiency (TPR)")
    plt.title("Cut-based ID ROC (Validation)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ROC_curve.png"), dpi=150)
    plt.close()

    # Summary Table
    table = Table(title=f"Cut-Based ID Summary")
    table.add_column("WP")
    table.add_column("Eff (val)")
    table.add_column("Bkg Rej (val)")
    for wp, d in WPs.items():
        table.add_row(wp, f"{d['eff_val']:.3f}", f"{d['rej_val']:.3f}")
    console.print(table)

    with open(os.path.join(outdir, "CutBased_WPs.json"), "w") as f:
        json.dump(WPs, f, indent=2)

    return {"WPs": WPs, "roc": os.path.join(outdir, "ROC_curve.png")}

def TrainCutBasedIDAnalysis(**kwargs):
    regions = kwargs.pop("regions", [])
    inputdir = kwargs.pop("inputdir")
    variables = kwargs.pop("variables")
    outdir = kwargs.pop("outputdir")
    n_trials = kwargs.pop("n_trials")

    for region in regions:
        artifacts = train_cutbased_ga(
            train_signal_path=os.path.join(inputdir, region, "train", "signal.parquet"),
            train_bkg_path=os.path.join(inputdir, region, "train", "background.parquet"),
            valid_signal_path=os.path.join(inputdir, region, "valid", "signal.parquet"),
            valid_bkg_path=os.path.join(inputdir, region, "valid", "background.parquet"),
            variables=variables,
            target_efficiencies=[0.90, 0.80, 0.70],
            outdir=os.path.join(outdir, region),
            n_trials=2  # Number of Optuna trials
        )

