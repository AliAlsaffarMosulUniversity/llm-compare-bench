"""Create the figures used in the paper from results/example_validation/*.csv."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def power_figure():
    rows = list(csv.DictReader(open(ROOT / "results/example_validation/experiment_b_power_grid.csv")))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    colors = {"0.0": "#6b7280", "0.03": "#2563eb", "0.05": "#d97706", "0.1": "#059669"}
    for ax, acc in zip(axes, ("0.8", "0.9")):
        for gap, c in colors.items():
            pts = sorted((int(r["n"]), float(r["rejection_rate"])) for r in rows
                         if r["acc_a"] == acc and r["gap"] == gap and r["rho"] == "0.5")
            label = "no true gap (type-I error)" if gap == "0.0" else f"true gap = {int(float(gap) * 100)} pp"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, lw=1.8, color=c, label=label,
                    ls="--" if gap == "0.0" else "-")
        ax.axhline(0.05, color="#9ca3af", lw=0.8, ls=":")
        ax.axhline(0.80, color="#9ca3af", lw=0.8, ls=":")
        ax.axvspan(150, 300, color="#fde68a", alpha=0.35, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 200, 300, 500, 1000])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Number of paired tasks (log scale)")
        ax.set_title(f"System A accuracy = {int(float(acc) * 100)}%", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Share of simulated studies\nrejecting H0 (exact McNemar, α = 0.05)")
    for ax in axes:
        ax.set_ylim(-0.02, 1.14)
        ax.text(158, 1.07, "150–300 tasks (protocol range)", fontsize=7.5, color="#92400e")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig_power_curves.png", dpi=200)
    plt.close(fig)


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 36); ax.axis("off")
    boxes = [
        (2, "Task set\n(JSONL, SHA-256\nrecorded)", "#dbeafe"),
        (22, "Runner\n(resumable, same\nprompt & params)", "#e0e7ff"),
        (42, "Provider\nadapters\nOpenAI · Gemini", "#fde68a"),
        (62, "Scorers\nMCQ · numeric · code\nneedle · abstain", "#d1fae5"),
        (82, "Paired statistics\nMcNemar · bootstrap\nHolm · κ", "#fecaca"),
    ]
    for x, text, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 14), 16, 14, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=c, ec="#374151", lw=1))
        ax.text(x + 8, 21, text, ha="center", va="center", fontsize=8)
    for x in (18.9, 38.9, 58.9, 78.9):
        ax.annotate("", xy=(x + 2.6, 21), xytext=(x, 21), arrowprops=dict(arrowstyle="->", color="#374151"))
    ax.add_patch(FancyBboxPatch((30, 2), 56, 6, boxstyle="round,pad=0.4,rounding_size=1.2", fc="#f3f4f6",
                                ec="#6b7280", lw=1, ls="--"))
    ax.text(58, 5, "Open-ended tasks → blind X/Y rating sheets → ≥3 raters → Fleiss' κ → unblind",
            ha="center", va="center", fontsize=7.8)
    ax.annotate("", xy=(70, 8.6), xytext=(70, 13.2), arrowprops=dict(arrowstyle="->", color="#6b7280", ls="--"))
    ax.text(50, 33, "Config file fixes model IDs, sampling parameters and run date; raw responses are stored verbatim",
            ha="center", fontsize=8, color="#374151")
    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    power_figure(); pipeline_figure()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
