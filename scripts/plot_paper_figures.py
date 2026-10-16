#!/usr/bin/env python3
"""Reproduce Figures 2 and 3 from their released source CSV files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "figures" / "source_data"
DEFAULT_OUTPUT = ROOT / "figures"

COLORS = {
    "uncorrected": "#F3C6A3",
    "cess": "#8EC5DA",
    "grid": "#E9E9E9",
    "text": "#000000",
}


def configure_style() -> None:
    """Use Times New Roman when installed, with portable serif fallbacks."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 12,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def style_axis(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.75)
    ax.tick_params(length=3.2, width=0.8, colors=COLORS["text"])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#000000")
        spine.set_linewidth(0.8)


def panel_label(ax: plt.Axes, text: str, y: float, fontsize: float) -> None:
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="normal",
        va="top",
        ha="center",
        color=COLORS["text"],
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 300, "facecolor": "white"}),
    ):
        fig.savefig(
            output_dir / f"{stem}{suffix}",
            bbox_inches="tight",
            pad_inches=0.045,
            **kwargs,
        )
    plt.close(fig)


def plot_ranking_sensitivity(source_dir: Path, output_dir: Path) -> None:
    rows = read_rows(source_dir / "fig_search_bias_and_stability.csv")
    lookup = {(r["panel"], r["setting"], r["method"]): float(r["conclusion_gap"]) for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.85), constrained_layout=True)

    controlled = [
        lookup[("a", "controlled", "uncorrected")],
        lookup[("a", "controlled", "causal_correction")],
    ]
    bars = axes[0].bar(
        ["Opened mean", "CESS correction"],
        controlled,
        width=0.44,
        color=[COLORS["uncorrected"], COLORS["cess"]],
        edgecolor="none",
    )
    axes[0].set_ylim(0, 0.82)
    axes[0].set_ylabel("Conclusion gap ↓", fontsize=20)
    style_axis(axes[0])
    axes[0].grid(False)
    axes[0].tick_params(axis="x", labelsize=20)
    axes[0].tick_params(axis="y", labelsize=17)
    for index, (bar, value) in enumerate(zip(bars, controlled)):
        y = value - 0.110 if index == 0 else 0.047
        axes[0].text(bar.get_x() + bar.get_width() / 2, y, f"{value:.4f}", ha="center", va="center", fontsize=17, color=COLORS["text"])
    reduction = 100 * (controlled[0] - controlled[1]) / controlled[0]
    axes[0].text(0.68, 0.53, f"{reduction:.1f}% smaller", transform=axes[0].transAxes, ha="center", va="center", fontsize=17, color=COLORS["text"], fontweight="bold")

    models = ["Qwen2.5-32B", "OLMo3-7B"]
    opened = np.array([lookup[("b", model, "uncorrected")] for model in models])
    cess = np.array([lookup[("b", model, "CESS")] for model in models])
    x = np.arange(len(models)) * 1.4
    width = 0.31
    bars_opened = axes[1].bar(x - 0.25, opened, width, color=COLORS["uncorrected"], edgecolor="none")
    bars_cess = axes[1].bar(x + 0.25, cess, width, color=COLORS["cess"], edgecolor="none")
    axes[1].set_xticks(x, models)
    axes[1].set_ylim(0, 0.067)
    axes[1].set_xlim(-0.65, 2.05)
    style_axis(axes[1])
    axes[1].grid(False)
    axes[1].tick_params(axis="x", labelsize=20)
    axes[1].tick_params(axis="y", labelsize=17)
    for bars_group in (bars_opened, bars_cess):
        for bar in bars_group:
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=17, color=COLORS["text"])
    axes[1].legend(
        [Patch(facecolor=COLORS["uncorrected"]), Patch(facecolor=COLORS["cess"])],
        ["Opened mean", "Task-wise CESS"],
        loc="upper right",
        frameon=False,
        handlelength=1.0,
        borderaxespad=0.25,
        prop={"family": "serif", "size": 20, "weight": "normal"},
    )
    panel_label(axes[0], "(a) Controlled ranking intervention", -0.16, 20)
    panel_label(axes[1], "(b) Real review tasks", -0.16, 20)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.015, hspace=0.015)
    save(fig, output_dir, "fig_search_bias_and_stability")


def plot_budget_efficiency(source_dir: Path, output_dir: Path) -> None:
    rows = read_rows(source_dir / "fig_budget_efficiency.csv")
    lookup = {(r["model"], int(r["search_rounds"]), r["method"]): float(r["mae"]) for r in rows}
    rounds = np.array([1, 2, 3, 4])
    models = ["Qwen2.5-32B", "OLMo3-7B"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.40), sharey=True, constrained_layout=True)
    for model, ax in zip(models, axes):
        opened = np.array([lookup[(model, int(r), "uncorrected")] for r in rounds])
        cess = np.array([lookup[(model, int(r), "Task-wise CESS")] for r in rounds])
        opened_line, = ax.plot(rounds, opened, marker="o", markersize=5.4, linewidth=2.0, color=COLORS["uncorrected"])
        cess_line, = ax.plot(rounds, cess, marker="s", markersize=5.4, linewidth=2.2, color=COLORS["cess"])
        ax.fill_between(rounds, cess, opened, color=COLORS["cess"], alpha=0.09, linewidth=0)
        ax.set_xticks(rounds)
        ax.set_xlim(0.85, 4.35)
        ax.set_xlabel("Search rounds", fontsize=15.5)
        ax.set_ylim(0.12, 0.435)
        style_axis(ax)
        ax.grid(False)
        ax.tick_params(labelsize=15.5)
        gains = 100 * (opened - cess) / opened
        for round_id, upper, lower, gain in zip(rounds, opened, cess, gains):
            gap = upper - lower
            ax.annotate("", xy=(round_id, lower + 0.12 * gap), xytext=(round_id, upper - 0.12 * gap), arrowprops={"arrowstyle": "-|>", "color": "#000000", "linewidth": 0.9, "mutation_scale": 9}, zorder=5)
            label_x = round_id + 0.25 if round_id == 1 else round_id
            label_y = {1: upper + 0.025, 2: upper + 0.043, 3: upper + 0.018, 4: upper + 0.016}[int(round_id)]
            ax.text(label_x, label_y, f"{gain:.1f}%", ha="center", va="bottom", fontsize=15.5, color=COLORS["text"])
        ax.legend([opened_line, cess_line], ["Opened mean", "Task-wise CESS"], loc="upper right", frameon=False, handlelength=1.4, prop={"family": "serif", "size": 15.5, "weight": "normal"})
    axes[0].set_ylabel("Mean absolute error ↓", fontsize=15.5)
    panel_label(axes[0], "(a) Qwen2.5-32B", -0.28, 15.5)
    panel_label(axes[1], "(b) OLMo3-7B", -0.28, 15.5)
    for text in fig.findobj(mpl.text.Text):
        text.set_fontsize(17)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.015, hspace=0.015)
    save(fig, output_dir, "fig_budget_efficiency")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    plot_ranking_sensitivity(args.source_dir, args.output_dir)
    plot_budget_efficiency(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
