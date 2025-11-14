import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
from rich.console import Console
from rich.table import Table
import matplotlib.colors as mc
import colorsys

def lighten_color(color, amount=0.3):
    """
    Lightens color by blending with white.
    amount=0 returns original color.
    amount=1 returns white.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    r, g, b = mc.to_rgb(c)
    return (1 - amount) * np.array([r, g, b]) + amount * np.array([1, 1, 1])


def shift_hue(color, shift=0.15):
    """
    Shift the color hue (for signal vs background separation).
    """
    r, g, b = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + shift) % 1.0
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)


plt.style.use(hep.style.CMS)
console = Console()

# -----------------------------
# Color-blind friendly palette
# -----------------------------
# Okabe–Ito palette (8 colors), safe for most color-vision deficiencies
OKABE_ITO = [
    "#ffa90e",
    "#3f90da",
    "#bd1f01",
    "#b9ac70",  # orange
    "#e76300",  # green
    "#a96b59",  # vermillion
    "#832db6",  # reddish purple
]

# Line styles to differentiate WPs and S/B
WP_LINESTYLES = ["-", "--", "-.", ":"]
SB_STYLES = {"signal": "-", "background": "--"}


@dataclass
class WP:
    name: str
    window_cuts: Dict[str, Tuple[float, float]]  # var -> (low, high)
    meta: Dict[str, float]  # e.g. {"eff_selected_from_tmva": 0.7}


# -----------------------------
# WP loading (TMVA-style)
# -----------------------------
def _normalize_window_cuts(entry: dict) -> Dict[str, Tuple[float, float]]:
    """
    Expect TMVA-like structure with 'window_cuts':
      "window_cuts": { "var": [low, high], ... }

    If only 'cuts' is present, interpret as (-inf, thr].
    """
    if "window_cuts" in entry:
        win = entry["window_cuts"]
        return {k: (float(v[0]), float(v[1])) for k, v in win.items()}
    elif "cuts" in entry:
        # Fallback: treat cuts as upper thresholds.
        return {k: (-1e30, float(v)) for k, v in entry["cuts"].items()}
    else:
        raise KeyError("No 'window_cuts' or 'cuts' in WP JSON entry")


def load_tmva_wp_file(path: str) -> List[WP]:
    with open(path, "r") as f:
        spec = json.load(f)

    wps: List[WP] = []
    for wp_name, cfg in spec.items():
        window_cuts = _normalize_window_cuts(cfg)
        meta = {k: v for k, v in cfg.items()
                if k not in ("cuts", "window_cuts")}
        wps.append(WP(name=wp_name, window_cuts=window_cuts, meta=meta))

    # Sort by eff_selected_from_tmva if present
    wps.sort(key=lambda w: w.meta.get("eff_selected_from_tmva", np.nan))
    return wps


# -----------------------------
# Core analysis class
# -----------------------------
class ValidationAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        wp_sources: Dict[str, List[WP]],
        class_col: str = "is_signal",
        region_col: str = "region",
        weight_col: str = "weight",
    ):
        self.df = df.copy()
        self.wp_sources = wp_sources
        self.class_col = class_col
        self.region_col = region_col
        self.weight_col = weight_col

        if self.weight_col not in self.df.columns:
            self.df[self.weight_col] = 1.0

        # Reweighting configuration
        self.reweight_vars: List[str] = []
        self.reweight_target_class: Optional[str] = None
        self.reweight_reference_class: Optional[str] = None
        self.reweight_maps_by_region: Dict[str, Tuple[np.ndarray, List[np.ndarray]]] = {}

    # ---------------------------
    # Masks & efficiencies
    # ---------------------------
    def _mask_class(self, is_signal: bool) -> np.ndarray:
        col = self.df[self.class_col]
        if col.dtype == bool:
            return col.to_numpy() == is_signal
        return (col.astype(int).to_numpy() == (1 if is_signal else 0))

    def _mask_region(self, region: str) -> np.ndarray:
        return self.df[self.region_col].astype(str).to_numpy() == str(region)

    @staticmethod
    def _apply_window_cuts_block(block: pd.DataFrame,
                                 window_cuts: Dict[str, Tuple[float, float]]) -> np.ndarray:
        mask = np.ones(len(block), dtype=bool)
        for var, (lo, hi) in window_cuts.items():
            if var not in block.columns:
                raise KeyError(f"Variable '{var}' not in dataframe")
            vals = block[var].to_numpy()
            mask &= (vals >= lo) & (vals <= hi)
        return mask

    def _pass_wp(self, df_block: pd.DataFrame, wp: WP) -> np.ndarray:
        return self._apply_window_cuts_block(df_block, wp.window_cuts)

    @staticmethod
    def _efficiency(sel: np.ndarray, weights: np.ndarray) -> float:
        num = float(np.sum(weights[sel]))
        den = float(np.sum(weights))
        return num / den if den > 0 else np.nan

    # ---------------------------
    # Kinematic reweighting
    # ---------------------------
    def enable_reweight(
        self,
        variables: List[str],
        target_class: str = "signal",
        reference_class: str = "background",
        bins: Tuple[int, ...] = (40, 40),
        ranges: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Histogram-based reweighting in each region: compute
        w(target)/w(reference) in (variables) space and apply to reference class.
        """
        assert target_class in ("signal", "background")
        assert reference_class in ("signal", "background")
        assert target_class != reference_class

        for v in variables:
            if v not in self.df.columns:
                raise KeyError(f"Reweighting variable '{v}' not found in dataframe")

        self.reweight_vars = variables
        self.reweight_target_class = target_class
        self.reweight_reference_class = reference_class
        self.reweight_maps_by_region.clear()

        regions = sorted(self.df[self.region_col].astype(str).unique())
        for reg in regions:
            reg_mask = self._mask_region(reg)
            if not reg_mask.any():
                continue

            reg_df = self.df.loc[reg_mask, :]
            x = reg_df[self.reweight_vars].to_numpy()
            w = reg_df[self.weight_col].to_numpy()

            col = self.df[self.class_col]
            if col.dtype == bool:
                is_sig_global = col.to_numpy()
            else:
                is_sig_global = (col.astype(int).to_numpy() == 1)

            is_sig = is_sig_global[reg_mask]

            def _cls_mask(name: str) -> np.ndarray:
                return is_sig if name == "signal" else (~is_sig)

            t_mask = _cls_mask(self.reweight_target_class)
            r_mask = _cls_mask(self.reweight_reference_class)

            H_t, edges = np.histogramdd(x[t_mask], bins=bins, range=ranges,
                                        weights=w[t_mask])
            H_r, _ = np.histogramdd(x[r_mask], bins=bins, range=ranges,
                                    weights=w[r_mask])

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(H_r > 0, H_t / H_r, 0.0)

            self.reweight_maps_by_region[reg] = (ratio, edges)

    def _lookup_reweight(self, reg: str, vec: np.ndarray) -> float:
        ratio, edges = self.reweight_maps_by_region[reg]
        idx = []
        for d, e in enumerate(edges):
            i = np.searchsorted(e, vec[d], side="right") - 1
            i = max(0, min(i, len(e) - 2))
            idx.append(i)
        return float(ratio[tuple(idx)])

    def _effective_weights(self, reg_mask: np.ndarray) -> np.ndarray:
        base_w = self.df.loc[reg_mask, self.weight_col].to_numpy().copy()
        if not self.reweight_vars:
            return base_w

        if not reg_mask.any():
            return base_w

        reg_name = str(self.df.loc[reg_mask, self.region_col].iloc[0])
        if reg_name not in self.reweight_maps_by_region:
            return base_w

        col = self.df[self.class_col]
        if col.dtype == bool:
            is_sig_global = col.to_numpy()
        else:
            is_sig_global = (col.astype(int).to_numpy() == 1)
        is_sig = is_sig_global[reg_mask]

        apply_to_signal = (self.reweight_reference_class == "signal")
        apply_mask = is_sig if apply_to_signal else (~is_sig)

        X = self.df.loc[reg_mask, self.reweight_vars].to_numpy()
        for i in range(len(base_w)):
            if apply_mask[i]:
                base_w[i] *= self._lookup_reweight(reg_name, X[i])
        return base_w

    # ---------------------------
    # Plotting helpers
    # ---------------------------
    @staticmethod
    def _prep_axes(ax: plt.Axes, grid: bool = True) -> None:
        if grid:
            ax.grid(True, alpha=0.3, linestyle=":")

    @staticmethod
    def _save(fig: plt.Figure, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    # ---------------------------
    # 1) ROC comparison
    # ---------------------------
    def plot_roc_by_region(self, outdir: str) -> None:
        regions = sorted(self.df[self.region_col].astype(str).unique())
        for reg in regions:
            reg_mask = self._mask_region(reg)
            if not reg_mask.any():
                continue

            eff_w = self._effective_weights(reg_mask)
            col = self.df[self.class_col]
            if col.dtype == bool:
                is_sig_global = col.to_numpy()
            else:
                is_sig_global = (col.astype(int).to_numpy() == 1)
            is_sig = is_sig_global[reg_mask]
            is_bkg = ~is_sig

            w_sig_all = eff_w[is_sig]
            w_bkg_all = eff_w[is_bkg]

            fig, ax = plt.subplots(figsize=(8, 8))
            # hep.cms.label("Preliminary", ax=ax, loc=0)

            colors = OKABE_ITO
            for s_i, (source, wps) in enumerate(self.wp_sources.items()):
                xs, ys = [], []
                for wp in wps:
                    block = self.df.loc[reg_mask, :]
                    pass_mask = self._pass_wp(block, wp)
                    sig_eff = self._efficiency(pass_mask[is_sig], w_sig_all)
                    bkg_eff = self._efficiency(pass_mask[is_bkg], w_bkg_all)
                    xs.append(bkg_eff)
                    ys.append(sig_eff)

                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    linestyle="-",
                    color=colors[s_i % len(colors)],
                    label=source,
                )

                for (x, y), wp in zip(zip(xs, ys), wps):
                    ax.annotate(
                        wp.name,
                        (x, y),
                        textcoords="offset points",
                        xytext=(5, -5),
                        fontsize=8,
                    )

            self._prep_axes(ax)
            ax.set_xlabel("Background efficiency")
            ax.set_ylabel("Signal efficiency")
            ax.text(
              0.05, 0.95,
              f"{reg}",
              transform=ax.transAxes,
              fontsize=18,
              fontweight="bold",
              va="top",
              ha="left",
            )
#            ax.set_xlim(0, 1)
#            ax.set_ylim(0, 1)
            ax.legend()

            out = os.path.join(outdir, f"roc_region-{reg}.png")
            self._save(fig, out)

    # ---------------------------
    # 2) Efficiency vs variable
    # ---------------------------
    def plot_efficiency_vs_variables(
        self,
        variables: Iterable[str],
        binning: Dict[str, List[float]],
        outdir: str,
    ) -> None:
        """
        binning: dict[var] -> either [min, max, nbins] or full edges.
        """
        regions = sorted(self.df[self.region_col].astype(str).unique())
        variables = list(variables)

        for v in variables:
            if v not in self.df.columns:
                raise KeyError(f"Variable '{v}' not found in dataframe")

        for reg in regions:
            reg_mask = self._mask_region(reg)
            if not reg_mask.any():
                continue

            eff_w = self._effective_weights(reg_mask)
            reg_df = self.df.loc[reg_mask, :]

            col = self.df[self.class_col]
            if col.dtype == bool:
                is_sig_global = col.to_numpy()
            else:
                is_sig_global = (col.astype(int).to_numpy() == 1)
            is_sig = is_sig_global[reg_mask]
            is_bkg = ~is_sig

            for var in variables:
                vals = reg_df[var].to_numpy()
                bins_cfg = binning[var]
                if len(bins_cfg) == 3:
                    vmin, vmax, nb = bins_cfg
                    b_edges = np.linspace(vmin, vmax, nb + 1)
                else:
                    b_edges = np.asarray(bins_cfg, dtype=float)

                centers = 0.5 * (b_edges[:-1] + b_edges[1:])
                fig, ax = plt.subplots(figsize=(10, 8))
#                hep.cms.label("Preliminary", ax=ax, loc=0)
                colors = OKABE_ITO

                for s_i, (source, wps) in enumerate(self.wp_sources.items()):
                    base_color = colors[s_i % len(colors)]

                    # SHADES FOR DIFFERENT WPs
                    n_wp = len(wps)
                    wp_shades = [
                        lighten_color(base_color, amount=0.1 + 0.6 * (i / max(n_wp - 1, 1)))
                        for i in range(n_wp)
                    ]

                    # SHIFT hue for signal/background separation
                    sig_color = base_color
                    bkg_color = shift_hue(base_color, shift=0.25)

                    for w_i, wp in enumerate(wps):
                        wp_color_sig = wp_shades[w_i]              # wp-based shade for signal
                        wp_color_bkg = shift_hue(wp_shades[w_i])   # hue-shifted shade for background

                        pass_mask = self._pass_wp(reg_df, wp)

                        for cls_name, cls_mask, cls_color in (
                            ("signal", is_sig, wp_color_sig),
                            ("background", is_bkg, wp_color_bkg),
                        ):
                            cls_w = eff_w[cls_mask]
                            cls_vals = vals[cls_mask]
                            sel = pass_mask[cls_mask]

                            tot, _ = np.histogram(cls_vals, bins=b_edges, weights=cls_w)
                            pas, _ = np.histogram(cls_vals[sel], bins=b_edges, weights=cls_w[sel])

                            with np.errstate(divide="ignore", invalid="ignore"):
                                eff = np.where(tot > 0, pas / tot, np.nan)
                                err = np.where(
                                    tot > 0,
                                    np.sqrt(eff * (1 - eff) / tot),
                                    np.nan,
                                )

                            linestyle = "none" # WP_LINESTYLES[w_i % len(WP_LINESTYLES)]
                            xerr = 0.5 * (b_edges[1:] - b_edges[:-1])

                            ax.errorbar(
                                centers,
                                eff,
                                xerr = xerr,
                                yerr=err,
                                fmt="o",
                                markersize=3,
                                linestyle=linestyle,
                                color=cls_color,
                                ecolor=cls_color,
                                capsize=2,
                                label=f"{source} {wp.name} {cls_name}",
                            )

                self._prep_axes(ax)
                ax.set_xlabel(var)
                ax.set_ylabel("Efficiency")
                ax.set_ylim(0, 1.1)
                ax.legend(fontsize=7, ncol=2, frameon=True)
                out = os.path.join(outdir, f"eff_{var}_region-{reg}.png")
                self._save(fig, out)

    # ---------------------------
    # 3) N−1 plots
    # ---------------------------
    def plot_nminus1_distributions(
        self,
        variables: Iterable[str],
        binning: Dict[str, List[float]],
        outdir: str,
        density: bool = True,
    ) -> None:
        """
        For each variable, WP, source:
          - Apply all cuts except that variable (N−1)
          - Plot S/B distributions and cut window
        """
        regions = sorted(self.df[self.region_col].astype(str).unique())
        variables = list(variables)

        for v in variables:
            if v not in self.df.columns:
                raise KeyError(f"Variable '{v}' not found in dataframe")

        for reg in regions:
            reg_mask = self._mask_region(reg)
            if not reg_mask.any():
                continue

            eff_w = self._effective_weights(reg_mask)
            reg_df = self.df.loc[reg_mask, :]

            col = self.df[self.class_col]
            if col.dtype == bool:
                is_sig_global = col.to_numpy()
            else:
                is_sig_global = (col.astype(int).to_numpy() == 1)
            is_sig = is_sig_global[reg_mask]
            is_bkg = ~is_sig

            for var in variables:
                vals = reg_df[var].to_numpy()
                bins_cfg = binning[var]
                if len(bins_cfg) == 3:
                    vmin, vmax, nb = bins_cfg
                    b_edges = np.linspace(vmin, vmax, nb + 1)
                else:
                    b_edges = np.asarray(bins_cfg, dtype=float)
                    vmin, vmax = float(b_edges[0]), float(b_edges[-1])
                centers = 0.5 * (b_edges[:-1] + b_edges[1:])

                for s_i, (source, wps) in enumerate(self.wp_sources.items()):
                    color = OKABE_ITO[s_i % len(OKABE_ITO)]
                    for wp in wps:
                        # N−1 cuts: remove 'var' from cuts
                        nm1_cuts = {k: v for k, v in wp.window_cuts.items()
                                    if k != var}
                        if nm1_cuts:
                            nm1_mask = self._apply_window_cuts_block(reg_df, nm1_cuts)
                        else:
                            nm1_mask = np.ones(len(reg_df), dtype=bool)

                        fig, ax = plt.subplots(figsize=(5.8, 4.6))
                        hep.cms.label("Preliminary", ax=ax, loc=0)

                        for cls_name, cls_mask, alpha in (
                            ("signal", is_sig, 0.5),
                            ("background", is_bkg, 0.5),
                        ):
                            use = nm1_mask & cls_mask
                            if not use.any():
                                continue
                            hist_w, _ = np.histogram(
                                vals[use],
                                bins=b_edges,
                                weights=eff_w[use],
                                density=density,
                            )
                            ax.step(
                                centers,
                                hist_w,
                                where="mid",
                                color=color,
                                linestyle=SB_STYLES[cls_name],
                                alpha=alpha,
                                label=f"{source} {wp.name} {cls_name}",
                            )

                        # Draw cut window if variable has a cut
                        if var in wp.window_cuts:
                            lo, hi = wp.window_cuts[var]
                            ax.axvline(lo, color="k", linestyle=":", linewidth=1)
                            ax.axvline(hi, color="k", linestyle=":", linewidth=1)
                            ax.fill_betweenx(
                                [0, ax.get_ylim()[1]],
                                lo,
                                hi,
                                color="grey",
                                alpha=0.1,
                            )

                        self._prep_axes(ax)
                        ax.set_xlabel(var)
                        ax.set_ylabel("Density" if density else "Counts")
                        ax.set_title(f"N−1: {var} — {source} {wp.name} — Region: {reg}")
                        ax.legend(fontsize=7, frameon=True)
                        out = os.path.join(
                            outdir,
                            f"nminus1_{var}_region-{reg}_{source}_{wp.name}.png",
                        )
                        self._save(fig, out)


# -----------------------------
# Main Luigi-facing entry
# -----------------------------
def ValidateIDAnalysis(**kwargs):
    """
    Validation of cut-based ID WPs across multiple sources and regions.

    Expected kwargs (from YAML via WorkflowManager):
      inputdir: base directory containing region subfolders with validation data
        inputdir/REGION/valid/signal.parquet
        inputdir/REGION/valid/background.parquet

      regions: list of region names (e.g. ["Barrel", "Endcap"])

      monitor_eff:
        var:
          binning: [edges...] or [min, max, nbins]

      variables:  (for N−1 plots)
        var:
          binning: [edges...] or [min, max, nbins]

      # Optional:
      wp_sources:
        SourceName: "/path/to/json/template_{region}.json"
        # {region} will be formatted with the region name.

      reweight_by: [pt, eta]   # (optional)
      reweight_target: "signal" or "background" (default: "signal")
      reweight_reference: "signal" or "background" (default: "background")
    """
    inputdir = kwargs.pop("inputdir")
    regions = kwargs.pop("regions")
    monitor_eff = kwargs.pop("monitor_eff")
    variables = kwargs.pop("variables")
    reweight_by = False


    outdir = kwargs.pop("outputdir", os.path.join(inputdir, "validation_plots"))
    wp_sources_cfg = kwargs.pop("wp_sources", None)

    os.makedirs(outdir, exist_ok=True)
    console.rule("[bold blue]ValidateID Analysis")

    # Summary table at the end
    table = Table(title="Validation Outputs")
    table.add_column("Region")
    table.add_column("Output directory")

    for region in regions:
        console.rule(f"[yellow]Region: {region}")
        region_dir = os.path.join(inputdir, region)

        # --- Load validation data ---
        sig_path = os.path.join(region_dir, "valid", "signal.parquet")
        bkg_path = os.path.join(region_dir, "valid", "background.parquet")
        if not os.path.exists(sig_path) or not os.path.exists(bkg_path):
            console.print(f"[red]Missing parquet files in {region_dir}[/red]")
            continue

        df_sig = pd.read_parquet(sig_path)
        df_bkg = pd.read_parquet(bkg_path)

        df_sig["is_signal"] = True
        df_bkg["is_signal"] = False
        df_sig["region"] = region
        df_bkg["region"] = region

        if "weight" not in df_sig.columns:
            df_sig["weight"] = 1.0
        if "weight" not in df_bkg.columns:
            df_bkg["weight"] = 1.0

        df = pd.concat([df_sig, df_bkg], ignore_index=True)

        # --- Resolve WP JSON paths for this region ---
        if wp_sources_cfg is not None:
            wp_paths: Dict[str, str] = {}
            for src_name, tmpl in wp_sources_cfg.items():
                path = tmpl.format(region=region)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"WP JSON for source '{src_name}' not found: {path}")
                wp_paths[src_name] = path
        else:
            # Default: single source named "Default" with JSON at REGION/CutBased_WPs.json
            default_path = os.path.join(region_dir, "CutBased_WPs.json")
            if not os.path.exists(default_path):
                raise FileNotFoundError(
                    "No 'wp_sources' specified and default CutBased_WPs.json "
                    f"not found at {default_path}"
                )
            wp_paths = {"Default": default_path}

        # --- Load TMVA-style WPs for all sources ---
        wp_sources: Dict[str, List[WP]] = {}
        for src_name, path in wp_paths.items():
            console.print(f"  [cyan]Source {src_name}[/cyan]: loading WPs from {path}")
            wp_sources[src_name] = load_tmva_wp_file(path)

        # --- Build analyzer ---
        va = ValidationAnalyzer(
            df,
            wp_sources,
            class_col="is_signal",
            region_col="region",
            weight_col="weight",
        )

        if reweight_by:
            console.print(
                f"[green]Applying kinematic reweighting"
            )
            va.enable_reweight(
                variables=reweight_by,
                target_class=rew_target,
                reference_class=rew_reference,
            )

        reg_out = os.path.join(outdir, region)

        # 1) ROC
        va.plot_roc_by_region(outdir=reg_out)

        # 2) Efficiency vs variables (monitor_eff)
        eff_binning = {v: cfg["binning"] for v, cfg in monitor_eff.items()}
        va.plot_efficiency_vs_variables(
            variables=monitor_eff.keys(),
            binning=eff_binning,
            outdir=os.path.join(reg_out, "eff_vs_var"),
        )

        # 3) N−1 plots (variables)
        nm1_binning = {v: cfg["binning"] for v, cfg in variables.items()}
        va.plot_nminus1_distributions(
            variables=variables.keys(),
            binning=nm1_binning,
            outdir=os.path.join(reg_out, "nminus1"),
            density=True,
        )

        console.print(f"✅ [bold green]Validation complete for {region}[/bold green]")
        table.add_row(region, reg_out)

    console.print(table)
    console.rule("[bold green]Validation Completed[/bold green]")

    return {"outputdir": outdir}

