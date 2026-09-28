from __future__ import annotations

"""A5c: redraw Fig. 3 / Fig. 5 / Fig. 6 with fold-level variability (R1#11)."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = WORKFLOW_ROOT.parent / "图" / "修订"
TAB = WORKFLOW_ROOT / "outputs" / "primary" / "tables"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.2,
    "font.size": 13,
    "axes.unicode_minus": False,
})

BLUE, RED = "#1f5fa8", "#cb4b3c"
C_BLUE, C_ORANGE, C_GREEN = "#1f77b4", "#d55e00", "#1b9e77"


def winsorize(vals: np.ndarray, lo: float = -1.0, hi: float = 1.05) -> np.ndarray:
    return np.clip(vals, lo, hi)


# Manuscript central values (user decision: keep original published numbers;
# variability overlays come from the reproduction run, differences are visually negligible)
FIG3_MS = {
    "XGBoost": (0.939, 0.649), "CatBoost": (0.904, 0.643), "ExtraTrees": (0.923, 0.631),
    "RandomForest": (0.907, 0.624), "LightGBM": (0.932, 0.622), "MLP": (0.910, 0.404),
}
FIG5_MS = {  # layer: (R2, RMSE, MAE)
    "Layer 1": (0.131, 1.609, 1.230), "Layer 2": (0.629, 1.052, 0.774),
    "L2+HOMO": (0.631, 1.049, 0.769), "L2+LUMO": (0.630, 1.050, 0.781),
    "Layer 3": (0.647, 1.026, 0.752),
}


# ---------------- Fig. 3 ----------------
def fig3() -> None:
    summ = pd.read_csv(TAB / "11_model_comparison_summary.csv")
    folds = pd.read_csv(TAB / "11_model_comparison_loso_fold_details.csv")
    order = ["XGBoost", "CatBoost", "ExtraTrees", "RandomForest", "LightGBM", "MLP"]
    summ = summ.set_index("model").loc[order].reset_index()
    # override central values with manuscript numbers
    summ["random_R2"] = [FIG3_MS[m][0] for m in order]
    summ["loso_R2"] = [FIG3_MS[m][1] for m in order]
    summ["GG"] = summ["random_R2"] - summ["loso_R2"]

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=300)
    x = np.arange(len(order))
    w = 0.36
    b1 = ax.bar(x - w / 2, summ["random_R2"], w, yerr=summ["random_R2_std"], capsize=4,
                color=BLUE, edgecolor="black", linewidth=0.8, label="Random split",
                error_kw=dict(lw=1.2, ecolor="black"))
    b2 = ax.bar(x + w / 2, summ["loso_R2"], w, color=RED, edgecolor="black", linewidth=0.8,
                label="Strict LOSO")
    for xi, v in zip(x - w / 2, summ["random_R2"]):
        ax.text(xi, v + 0.055, f"{v:.3f}", ha="center", fontsize=15, color=BLUE, fontweight="bold")
    # per-soil LOSO fold points: folds with R2 < 0 are stacked at the zero line
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        fr = folds.loc[folds["model"] == m, "r2"].to_numpy()
        fr_w = np.clip(fr, 0.0, 1.05)
        jitter = rng.uniform(-0.055, 0.055, size=len(fr_w))
        xpos = x[i] + w / 2 + 0.135
        ax.scatter(np.full(len(fr_w), xpos) + jitter, fr_w, s=13, color="gray",
                   alpha=0.4, zorder=3, linewidths=0)
        ax.text(x[i] + w / 2 - 0.06, summ.loc[i, "loso_R2"] + 0.035, f"{summ.loc[i, 'loso_R2']:.3f}",
                ha="center", fontsize=15, color=RED, fontweight="bold")
    # GG brackets
    for i, row in summ.iterrows():
        top = max(row["random_R2"], row["loso_R2"]) + 0.10
        ax.plot([x[i] - w / 2, x[i] + w / 2], [top, top], color="black", lw=1.1)
        ax.plot([x[i] - w / 2, x[i] - w / 2], [top, top - 0.02], color="black", lw=1.1)
        ax.plot([x[i] + w / 2, x[i] + w / 2], [top, top - 0.02], color="black", lw=1.1)
        ax.text(x[i], top + 0.02, f"GG={row['GG']:.3f}", ha="center", fontsize=14.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel(r"$R^2$", fontsize=17)
    ax.set_ylim(-0.05, 1.18)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.axhline(0, color="black", lw=0.9)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=16)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig3_revised.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig3_revised.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Fig3 saved")


# ---------------- Fig. 5 ----------------
def fig5() -> None:
    summ = pd.read_csv(TAB / "12_layer_ablation5_summary.csv")
    folds = pd.read_csv(TAB / "12_layer_ablation5_loso_fold_details.csv")
    order = ["Layer 1", "Layer 2", "L2+HOMO", "L2+LUMO", "Layer 3"]
    labels = ["Layer 1", "Layer 2", "L2\n+HOMO", "L2\n+LUMO", "Layer 3"]
    summ = summ.set_index("layer").loc[order].reset_index()
    # override central values with manuscript numbers
    summ["loso_R2"] = [FIG5_MS[l][0] for l in order]
    summ["loso_RMSE"] = [FIG5_MS[l][1] for l in order]
    summ["loso_MAE"] = [FIG5_MS[l][2] for l in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 6.0), dpi=300,
                                   gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.44))
    x = np.arange(len(order))
    ax1b = ax1.twinx()
    ax1.plot(x, summ["loso_R2"], "-o", color=C_BLUE, lw=2.4, ms=9, label=r"$R^2$", zorder=3)
    ax1b.plot(x, summ["loso_RMSE"], "--s", color=C_ORANGE, lw=2.4, ms=9, label="RMSE", zorder=3)
    ax1b.plot(x, summ["loso_MAE"], ":D", color=C_GREEN, lw=2.4, ms=9, label="MAE", zorder=3)
    for xi, v in zip(x, summ["loso_R2"]):
        ax1.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=15, color=C_BLUE, fontweight="bold")
    for xi, v in zip(x, summ["loso_RMSE"]):
        ax1b.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 13),
                      ha="center", fontsize=15, color=C_ORANGE, fontweight="bold")
    for xi, v in zip(x, summ["loso_MAE"]):
        ax1b.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, -22),
                      ha="center", fontsize=15, color=C_GREEN, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=15)
    ax1.tick_params(axis="y", labelsize=14)
    ax1b.tick_params(axis="y", labelsize=14)
    ax1.set_ylabel(r"Strict LOSO $R^2$", fontsize=16)
    ax1b.set_ylabel("RMSE / MAE", fontsize=16)
    ax1.set_ylim(0, 0.80)
    ax1b.set_ylim(0.66, 1.78)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3,
               frameon=False, fontsize=14.5)
    for s in ("top",):
        ax1.spines[s].set_visible(False)
        ax1b.spines[s].set_visible(False)
    ax1.text(-0.10, 1.05, "a", transform=ax1.transAxes, fontsize=22, fontweight="bold")

    # panel b: per-fold ΔR² distribution (boxplot) + pooled deltas as markers
    base = summ.loc[summ["layer"] == "Layer 2"].iloc[0]
    comp = ["L2+HOMO", "L2+LUMO", "Layer 3"]
    clabels = ["L2\n+HOMO", "L2+LUMO", "Layer 3"]
    xb = np.arange(len(comp))
    f_wide = folds.pivot_table(index="fold_id", columns="layer", values="r2")
    # pooled markers/text use manuscript values (FIG5_MS); boxplots use reproduction fold data
    MS_POOLED_DR2 = {"L2+HOMO": 0.002, "L2+LUMO": 0.001, "Layer 3": 0.018}
    MS_REL = {  # relative RMSE/MAE reduction vs Layer 2 (improvement, manuscript convention)
        "L2+HOMO": (0.003, 0.006), "L2+LUMO": (0.002, -0.009), "Layer 3": (0.025, 0.028),
    }
    data, pooled = [], []
    for c in comp:
        d = (f_wide[c] - f_wide["Layer 2"]).dropna().to_numpy()
        data.append(np.clip(d, -1.0, 1.0))
        pooled.append(MS_POOLED_DR2[c])
    bp = ax2.boxplot(data, positions=xb, widths=0.40, showfliers=True, patch_artist=True,
                     flierprops=dict(marker="o", ms=2.0, mfc="#c8c8c8", mec="none", alpha=0.30),
                     medianprops=dict(color="#333333", lw=1.6),
                     boxprops=dict(lw=1.0, color="#5a6b7d"),
                     whiskerprops=dict(lw=1.0, color="#5a6b7d"),
                     capprops=dict(lw=1.0, color="#5a6b7d"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbe6f3")
        patch.set_alpha(0.95)
    for i, c in enumerate(comp):
        ax2.plot(xb[i] + 0.30, pooled[i], marker="D", ms=9, color=C_ORANGE,
                 markeredgecolor="black", zorder=4)
        ax2.annotate(f"{pooled[i]:+.3f}", (xb[i] + 0.30, pooled[i]),
                     textcoords="offset points", xytext=(0, 12), ha="center", fontsize=14.5,
                     color=C_ORANGE, fontweight="bold")
    ax2.plot([], [], marker="D", ms=9, ls="none", color=C_ORANGE, markeredgecolor="black",
             label=r"Pooled $\Delta R^2$")
    ax2.legend(loc="upper right", frameon=False, fontsize=14)
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(clabels, fontsize=15)
    ax2.tick_params(axis="y", labelsize=14)
    ax2.set_ylabel(r"Per-fold $\Delta R^2$ vs Layer 2", fontsize=16)
    ax2.set_ylim(-1.05, 1.05)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.text(-0.10, 1.05, "b", transform=ax2.transAxes, fontsize=22, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig5_revised.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig5 saved")


# ---------------- Fig. 6 ----------------
def fig6() -> None:
    shap_df = pd.read_csv(TAB / "12_shap_per_fold.csv")
    top5 = pd.read_csv(TAB / "12_shap_top5_frequency.csv")
    layers = [("a", "Layer 1"), ("b", "Layer 2"), ("c", "Layer 3")]

    # Route B: per-fold attribution share (mean|SHAP| normalized to 100% within each
    # fold), then mean share across folds. Removes cross-fold scale artifact.
    # No error bars: cross-fold variability is conveyed by Top-5 frequency coloring
    # (panel c) and rank statistics reported in the text.
    rows = []
    for lname in shap_df["layer"].unique():
        piv = shap_df[shap_df["layer"] == lname].pivot_table(
            index="fold_id", columns="feature", values="mean_abs_shap")
        share = piv.div(piv.sum(axis=1), axis=0) * 100.0
        st = share.agg(["mean", "std"]).T.reset_index()
        st.columns = ["feature", "share_mean", "share_std"]
        st["layer"] = lname
        rows.append(st)
    stats = pd.concat(rows, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=300)
    cmap = plt.get_cmap("YlGnBu")
    xmax = 50
    for ax, (tag, lname) in zip(axes, layers):
        st = stats[stats["layer"] == lname].sort_values("share_mean", ascending=True)
        y = np.arange(len(st))
        if lname == "Layer 3":
            freq = top5[top5["layer"] == lname].set_index("feature")["top5_count"]
            counts = st["feature"].map(freq).fillna(0).to_numpy()
            colors = cmap(counts / 142.0)
            bars = ax.barh(y, st["share_mean"], color=colors,
                           edgecolor="black", lw=0.5)
        else:
            ax.barh(y, st["share_mean"], color="#b9c6d8",
                    edgecolor="none")
        ax.set_yticks(y)
        ax.set_yticklabels(st["feature"], fontsize=12)
        ax.set_xlim(0, xmax)
        ax.set_xticks(np.arange(0, xmax + 1, 15))
        ax.set_xlabel("Attribution share (%)", fontsize=12.5)
        ax.set_title(lname, fontsize=14)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(-0.22, 1.08, tag, transform=ax.transAxes, fontsize=19, fontweight="bold")
    # colorbar for panel c
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 142))
    cbar = fig.colorbar(sm, ax=axes[-1], pad=0.02, fraction=0.05)
    cbar.set_label("Top-5 frequency across 142 LOSO folds", fontsize=12)
    cbar.set_ticks([0, 35, 71, 106, 142])
    cbar.set_ticklabels(["0", "35", "71", "106", "142"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig6_revised.png", bbox_inches="tight")
    plt.close(fig)
    print("Fig6 saved")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig3()
    fig5()
    fig6()
    print("13 done.")


if __name__ == "__main__":
    main()
