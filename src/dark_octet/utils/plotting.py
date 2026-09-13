from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
 
BLUE = "#003087"
RED  = "#8B0000"
GOLD = "#B8860B"
GRAY = "#555555"
 
 
def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          11,
        "axes.labelsize":     13,
        "axes.titlesize":     13,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         150,
        "figure.autolayout":  True,
        "lines.linewidth":    2.0,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.4,
    })
 
 
def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)