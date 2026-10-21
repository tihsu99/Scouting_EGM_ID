import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import matplotlib.colors as mc
import colorsys
from workflow.data import edges, efficiency_stats
from workflow.InspectDistribution import define_variables
from workflow.plotting import add_legend, configure_style, save_plot, series_panels

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
        label: str = "",
        plot_style: Optional[dict] = None,
        processes: Optional[dict] = None,
        connect_sources: Iterable[str] = (),
        dedicated_wps: bool = False,
        regions: Optional[List[str]] = None,
    ):
        self.plot_style = plot_style
        configure_style(plot_style)
        # Only a missing weight column is added; keep large evaluation tables shared.
        self.df = df.copy(deep=False)
        self.label = label
        self.wp_sources = wp_sources
        self.class_col = class_col
        self.region_col = region_col
        self.weight_col = weight_col
        self.processes = processes
        self.connect_sources = set(connect_sources)
        self.dedicated_wps = dedicated_wps
        self.regions = regions or sorted(self.df[region_col].astype(str).unique())

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
        if wp.meta.get("strict", False):
            mask = np.ones(len(df_block), dtype=bool)
            for var, (lo, hi) in wp.window_cuts.items():
                mask &= (df_block[var].to_numpy() > lo) & (df_block[var].to_numpy() < hi)
            return mask
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
    def _save(self, fig: plt.Figure, path: str) -> None:
        save_plot(fig, path, self.plot_style)

    def _categories(self, block):
        if self.processes is None:
            return {name: block[self.class_col].astype(int).to_numpy() == int(signal)
                    for name, signal in (("signal", True), ("background", False))}
        return {name: (block["process"] == name).to_numpy() for name in self.processes}

    def _category_label(self, name):
        return self.processes[name].get("label", name) if self.processes else name

    def _is_signal(self, name):
        return self.processes[name]["is_signal"] if self.processes else name == "signal"

    def _wp_label(self, source, wp):
        template = (self.plot_style or {}).get("wp_label_format", "{source} {label}")
        return template.format(source=source, name=wp.name, label=wp.meta.get("label", wp.name))

    def _wp_styles(self, category=None):
        markers = ["o", "s", "^", "D", "v", "<", ">", "h", "p", "*", "X"]
        source_markers = (self.plot_style or {}).get("source_markers", {})
        pairs = [(source, wp) for source, wps in self.wp_sources.items() for wp in wps]
        palettes = (self.plot_style or {}).get("process_palettes", {})
        palette = (self.plot_style or {}).get("wp_palette")
        colors = palettes.get(category, palette if palette is not None else OKABE_ITO)
        if category not in palettes and category is not None and not self._is_signal(category):
            colors = [shift_hue(color) for color in colors]
        configured = category in palettes or palette is not None
        if configured and len(colors) < len(pairs):
            raise ValueError(f"Palette for {category} needs at least {len(pairs)} distinct WP colors")
        if configured and len({mc.to_rgba(color) for color in colors[:len(pairs)]}) != len(pairs):
            raise ValueError(f"Palette for {category} contains duplicate WP colors")
        return {(source, wp.name): (colors[i % len(colors)], source_markers.get(source, markers[i % len(markers)]))
                for i, (source, wp) in enumerate(pairs)}

    # ---------------------------
    # 1) ROC comparison
    # ---------------------------
    def plot_roc_by_region(self, outdir: str, filename_prefix: str = "") -> None:
        for reg in self.regions:
            reg_mask = self._mask_region(reg)
            block = self.df if reg_mask.all() else self.df.loc[reg_mask]
            weights = self._effective_weights(reg_mask)
            categories = self._categories(block)
            signal = next(name for name in categories if self._is_signal(name))
            backgrounds = [name for name in categories if not self._is_signal(name)]
            fig, ax = plt.subplots()
            if (self.plot_style or {}).get("roc_box_aspect") is not None:
                ax.set_box_aspect(self.plot_style["roc_box_aspect"])
            rows, annotations = [], {}
            for source, wps in self.wp_sources.items():
                selections = [self._pass_wp(block, wp) for wp in wps]
                for b_i, background in enumerate(backgrounds):
                    xs, ys = [], []
                    styles = self._wp_styles(background)
                    line_color = styles[(source, wps[0].name)][0]
                    for wp, passed in zip(wps, selections):
                        sig_mask, bkg_mask = categories[signal], categories[background]
                        sig_eff, sig_err, sig_neff = efficiency_stats(passed[sig_mask], weights[sig_mask])
                        bkg_eff, bkg_err, bkg_neff = efficiency_stats(passed[bkg_mask], weights[bkg_mask])
                        rows.append(dict(source=source, wp=wp.name, signal=signal, background=background,
                                         signal_efficiency=sig_eff, fake_rate=bkg_eff,
                                         signal_error=sig_err, fake_error=bkg_err,
                                         signal_neff=sig_neff, background_neff=bkg_neff,
                                         signal_rows=int(sig_mask.sum()), background_rows=int(bkg_mask.sum()),
                                         signal_pass_rows=int((passed & sig_mask).sum()),
                                         background_pass_rows=int((passed & bkg_mask).sum())))
                        xs.append(bkg_eff)
                        ys.append(sig_eff)
                        if np.isfinite(sig_eff) and np.isfinite(bkg_eff):
                            name = wp.meta.get("label", wp.name)
                            color, marker = styles[(source, wp.name)]
                            ax.plot(bkg_eff, sig_eff, marker=marker,
                                    linestyle="none", color=color, markersize=8, fillstyle="full",
                                    markerfacecolor=color, zorder=3,
                                    label=(f"{self._wp_label(source, wp)} / {background}" if self.processes
                                           else source if wp is wps[0] else "_nolegend_"))
                            if not self.processes:
                                annotations.setdefault((bkg_eff, sig_eff), []).append(name)
                    ax.plot(xs, ys, color=line_color, marker=None, zorder=2,
                            linestyle=("-" if b_i % 2 == 0 else "--") if source in self.connect_sources else "none",
                            label="_nolegend_")
            for (x, y), names in annotations.items():
                ax.annotate("\n".join(dict.fromkeys(names)), (x, y), textcoords="offset points",
                            xytext=(5, -7 if y > .85 else 5),
                            va="top" if y > .85 else "bottom", fontsize="x-small")
            ax.set(xlabel="Background efficiency", ylabel="Signal efficiency")
            title = reg + (f"\n{self.label}" if self.label else "")
            ax.set_title(title, fontsize="small", pad=35)
            missing = [self._category_label(name) for name, mask in categories.items() if not mask.any()]
            if missing:
                ax.text(.04, .08, "No entries: " + ", ".join(missing), transform=ax.transAxes, fontsize="small")
            ax.margins(x=.15, y=.2)
            add_legend(ax, self.plot_style)
            self._save(fig, os.path.join(outdir, f"{filename_prefix}roc_region-{reg}.png"))
            pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{filename_prefix}roc_region-{reg}.csv"), index=False)

    # ---------------------------
    # 2) Efficiency vs variable
    # ---------------------------
    def plot_efficiency_vs_variables(self, variables: Iterable[str], binning: Dict[str, List[float]],
                                     outdir: str, labels: Optional[Dict[str, str]] = None,
                                     xscales: Optional[Dict[str, str]] = None) -> None:
        labels = labels or {}
        per_process = (self.plot_style or {}).get("efficiency_per_process", False)
        x_errors = (self.plot_style or {}).get("efficiency_x_errors", not self.processes)
        for reg in self.regions:
            reg_mask = self._mask_region(reg)
            block = self.df if reg_mask.all() else self.df.loc[reg_mask]
            weights = self._effective_weights(reg_mask)
            categories = self._categories(block)
            styles = {category: self._wp_styles(None if per_process else category) for category in categories}
            for var in variables:
                b_edges = edges(binning[var])
                xscale = (xscales or {}).get(var, "linear")
                if xscale == "log" and (b_edges <= 0).any():
                    raise ValueError(f"Logarithmic {var} bins must be positive")
                rows = []
                for source, wps in self.wp_sources.items():
                    for wp in wps:
                        passed = self._pass_wp(block, wp)
                        for category, mask in categories.items():
                            values, cls_w, selected = block.loc[mask, var].to_numpy(), weights[mask], passed[mask]
                            for index, (low, high) in enumerate(zip(b_edges[:-1], b_edges[1:])):
                                inside = (values >= low) & ((values < high) if index < len(b_edges)-2 else (values <= high))
                                value, error, neff = efficiency_stats(selected[inside], cls_w[inside])
                                rows.append(dict(source=source, wp=wp.name, category=category,
                                                 low=low, high=high, efficiency=value, error=error,
                                                 effective_entries=neff, rows=int(inside.sum()),
                                                 pass_rows=int(selected[inside].sum()),
                                                 sumw=float(cls_w[inside].sum()),
                                                 pass_sumw=float(cls_w[inside][selected[inside]].sum()),
                                                 sumw2=float(np.square(cls_w[inside]).sum())))
                metrics = pd.DataFrame(rows)
                os.makedirs(outdir, exist_ok=True)
                metrics.to_csv(os.path.join(outdir, f"eff_region-{reg}_{var}.csv"), index=False)

                def draw(data, filename, ylabel="Efficiency", title=""):
                    groups = list(data.groupby(["source", "wp", "category"], sort=False))
                    fig, axes, panel_size = series_panels(len(groups), self.plot_style)
                    for index, ((source, name, category), points) in enumerate(groups):
                        axis = axes[index // panel_size]
                        color, marker = styles[category][(source, name)]
                        wp = next(wp for wp in self.wp_sources[source] if wp.name == name)
                        # Offsets separate coincident entries without changing bin membership or CSV values.
                        offset = (.65 * ((index % panel_size + .5) / min(len(groups), panel_size) - .5)
                                  if self.processes and not x_errors else 0.)
                        centers = (points.low + points.high) / 2 + offset * (points.high - points.low)
                        if xscale == "log":
                            centers = np.exp((np.log(points.low) + np.log(points.high))/2
                                             + offset * np.log(points.high / points.low))
                        axis.errorbar(centers, points.efficiency, yerr=points.error,
                                      xerr=[centers-points.low, points.high-centers] if x_errors else None,
                                      fmt=marker, linestyle="none", markersize=5,
                                      color=color, markerfacecolor=color,
                                      fillstyle="full",
                                      markeredgewidth=1.4, capsize=2,
                                      label=(self._wp_label(source, wp) if per_process else
                                             f"{self._wp_label(source, wp)} {self._category_label(category)}"))
                    for axis in axes:
                        axis.set(xlabel=labels.get(var, var), ylabel=ylabel, ylim=(0, 1.1), xscale=xscale)
                        axis.set_title(reg + (f" | {title}" if title else "")
                                       + (f"\n{self.label}" if self.label and not per_process else ""), fontsize="small", pad=35)
                        if self.processes and not x_errors:
                            axis.text(0, -.10, "Markers offset within bins", transform=axis.transAxes, fontsize="x-small")
                        add_legend(axis, self.plot_style, ncol=2)
                    self._save(fig, os.path.join(outdir, filename))

                def draw_processes(data, directory="", title=""):
                    for category in categories:
                        draw(data.loc[data.category == category],
                             os.path.join(directory, f"{category}_{var}_region-{reg}.png"),
                             ylabel="Signal efficiency" if self._is_signal(category) else "Background efficiency",
                             title=category + (f" | {title}" if title else ""))

                if per_process:
                    draw_processes(metrics)
                else:
                    draw(metrics, f"eff_{var}_region-{reg}.png")
                    for signal, category, ylabel in ((True, "signal", "Signal efficiency"), (False, "background", "Background efficiency")):
                        selected = metrics.category.map(self._is_signal) == signal
                        draw(metrics.loc[selected], f"{category}_{var}_region-{reg}.png", ylabel)
                if self.dedicated_wps:
                    for (source, name), points in metrics.groupby(["source", "wp"], sort=False):
                        wp = next(wp for wp in self.wp_sources[source] if wp.name == name)
                        if per_process:
                            draw_processes(points, f"per_wp/{source}/{name}", self._wp_label(source, wp))
                        else:
                            draw(points, f"per_wp/{source}/{name}/eff_{var}_region-{reg}.png",
                                 title=self._wp_label(source, wp))

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
                            nm1_mask = self._pass_wp(reg_df, WP(wp.name, nm1_cuts, wp.meta))
                        else:
                            nm1_mask = np.ones(len(reg_df), dtype=bool)

                        fig, ax = plt.subplots()

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
                            ax.axvspan(
                                np.clip(lo, b_edges[0], b_edges[-1]),
                                np.clip(hi, b_edges[0], b_edges[-1]),
                                color="grey",
                                alpha=0.1,
                            )

                        # Unbounded cut sentinels must not expand the displayed range.
                        ax.set_xlim(b_edges[0], b_edges[-1])
                        ax.set_xlabel(var)
                        ax.set_ylabel("Density" if density else "Counts")
                        add_legend(ax, self.plot_style, title=f"{reg}, N−1")
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

      datasets: optional named processes with path, is_signal, requirement and label
      define_variable: additional validation features, such as raw I/E for defaults
      connect_sources: sources whose discrete ROC points should be connected
      dedicated_wps: also export one efficiency figure per source/WP
    """
    inputdir = kwargs.pop("inputdir")
    regions = kwargs.pop("regions")
    monitor_eff = kwargs.pop("monitor_eff")
    variables = kwargs.pop("variables")
    pt_regions = kwargs.pop("pt_regions", {})
    requirement = kwargs.pop("requirement", None)
    datasets = kwargs.get("datasets")
    if datasets and sum(bool(spec["is_signal"]) for spec in datasets.values()) != 1:
        raise ValueError("Validation datasets must contain exactly one signal process")

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
        inputs = datasets or {name: {"path": os.path.join(region_dir, "valid", f"{name}.parquet"),
                                    "is_signal": name == "signal"} for name in ("signal", "background")}
        parts = []
        for name, spec in inputs.items():
            paths = spec["path"] if isinstance(spec["path"], (list, tuple)) else [spec["path"]]
            paths = [path.format(region=region) for path in paths]
            # Older PyArrow cannot infer the filesystem from a one-item path list.
            frame = pd.read_parquet(paths[0] if len(paths) == 1 else paths,
                                    columns=kwargs.get("input_columns"))
            if spec.get("requirement"):
                frame = frame.query(spec["requirement"])
            if kwargs.get("region_definition"):
                frame = frame.query(kwargs["region_definition"][region]["requirement"])
            if frame.empty:
                raise ValueError(f"No validation candidates for {region}/{name}")
            frame = frame.assign(is_signal=bool(spec["is_signal"]), region=region, process=name)
            if kwargs.get("unit_weights", False) or "weight" not in frame:
                frame["weight"] = 1.0
            parts.append(frame)
        df = pd.concat(parts, ignore_index=True)
        del parts, frame
        define_variables(df, kwargs.get("define_variable", {}))
        if requirement:
            df = df.query(requirement).copy()
        if df.empty:
            raise ValueError(f"No validation candidates for {region}")

        # --- Resolve WP JSON paths for this region ---
        if wp_sources_cfg is not None:
            wp_paths: Dict[str, str] = {}
            for src_name, tmpl in wp_sources_cfg.items():
                if not isinstance(tmpl, str):
                    tmpl = tmpl["path"]
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
            options = (wp_sources_cfg or {}).get(src_name, {})
            if not isinstance(options, str):
                for wp in wp_sources[src_name]:
                    expected = options.get("isolation_denominator")
                    if expected is not None and wp.meta.get("isolation_denominator") != expected:
                        raise ValueError(f"Isolation denominator mismatch in {path}/{wp.name}: expected {expected}")
                    mapping = options.get("variable_map", {})
                    wp.window_cuts = {mapping.get(var, var): cut for var, cut in wp.window_cuts.items()}
                    wp.meta["label"] = options.get("labels", {}).get(wp.name, wp.name)
            if not wp_sources[src_name]:
                raise ValueError(f"No working points in {path}")

        options = dict(processes=datasets, connect_sources=kwargs.get("connect_sources", ()),
                       dedicated_wps=kwargs.get("dedicated_wps", False), regions=[region],
                       plot_style=kwargs.get("plot_style"))

        # --- Build analyzer ---
        va = ValidationAnalyzer(
            df,
            wp_sources,
            class_col="is_signal",
            region_col="region",
            weight_col="weight",
            label=kwargs.get("label", ""),
            **options,
        )

        reg_out = os.path.join(outdir, region)

        # 1) ROC
        plot_style = kwargs.get("plot_style") or {}
        roc_out = plot_style.get("roc_outputdir", reg_out)
        roc_prefix = plot_style.get("roc_filename_prefix", "")
        va.plot_roc_by_region(outdir=roc_out, filename_prefix=roc_prefix)
        for label, (low, high) in pt_regions.items():
            subset = df.loc[(df.pt >= low) & (True if high is None else df.pt < high)]
            if subset.empty and not kwargs.get("allow_empty_slices", False):
                raise ValueError(f"No validation entries in {region}/{label}")
            pt_label = rf"$p_T \geq {low}$ GeV" if high is None else rf"${low} \leq p_T < {high}$ GeV"
            ValidationAnalyzer(subset, wp_sources, label=pt_label + "\n" + kwargs.get("label", ""),
                               **options).plot_roc_by_region(
                outdir=roc_out if "roc_outputdir" in plot_style else os.path.join(reg_out, label),
                filename_prefix=f"{roc_prefix}{label}_" if "roc_outputdir" in plot_style else roc_prefix)
            del subset

        # 2) Efficiency vs variables (monitor_eff)
        eff_binning = {v: cfg["binning"] for v, cfg in monitor_eff.items()}
        va.plot_efficiency_vs_variables(
            variables=monitor_eff.keys(),
            binning=eff_binning,
            outdir=os.path.join(reg_out, "eff_vs_var"),
            labels={v: cfg.get("label", v) for v, cfg in monitor_eff.items()},
            xscales={v: cfg.get("xscale", "linear") for v, cfg in monitor_eff.items()},
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
        del va, df

    console.print(table)
    console.rule("[bold green]Validation Completed[/bold green]")

    return {"outputdir": outdir}
