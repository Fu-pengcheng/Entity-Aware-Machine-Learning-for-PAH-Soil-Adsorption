from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


def apply_basic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

