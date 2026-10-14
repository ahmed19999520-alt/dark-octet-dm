import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .gmo_formula import octet_spectrum, mass_splittings
from ..constants import OCTET_QN, OCTET_ORDER


LATEX_LABELS = {
    "chi0":    r"$\chi^0$",
    "chi_p":   r"$\chi^+$",
    "chi_m":   r"$\chi^-$",
    "chi_pp":  r"$\chi^{++}$",
    "Sigma0":  r"$\Sigma^0_d$",
    "Sigma_m": r"$\Sigma^-_d$",
    "Sigma_p": r"$\Sigma^+_d$",
    "Lambda":  r"$\Lambda_d$",
}

I3_MAP = {
    "chi0":    +0.5,
    "chi_p":   -0.5,
    "chi_m":   +1.5,
    "chi_pp":  -1.5,
    "Sigma0":  0.0,
    "Sigma_m": +1.0,
    "Sigma_p": -1.0,
    "Lambda":  0.0,
}


def weight_diagram(save_path: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlabel(r"$I_{3,\mathrm{dark}}$", fontsize=13)
    ax.set_ylabel(r"$Y_{\mathrm{dark}}$", fontsize=13)
    ax.set_title(r"Dark-sector baryon octet weight diagram", fontsize=13)
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.axvline(0, color="gray", lw=0.6, ls="--")

    for name in OCTET_ORDER:
        qn   = OCTET_QN[name]
        i3   = I3_MAP[name]
        y    = qn["Y"]
        is_dm = name == "chi0"
        color = "gold" if is_dm else "steelblue"
        ax.scatter(i3, y, s=180, color=color, zorder=5,
                   edgecolors="black", linewidths=0.8)
        ax.annotate(
            LATEX_LABELS[name],
            xy=(i3, y),
            xytext=(i3 + 0.08, y + 0.07),
            fontsize=11,
        )

    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-1.4, 1.7)
    ax.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ax.set_yticks([-1, 0, 1])

    dm_patch  = mpatches.Patch(color="gold",     label=r"Stable DM $\chi^0$")
    un_patch  = mpatches.Patch(color="steelblue", label="Unstable (decays to $\\chi^0$)")
    ax.legend(handles=[dm_patch, un_patch], fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def bar_spectrum(save_path: str = None) -> plt.Figure:
    masses = octet_spectrum()
    splits = mass_splittings(masses)
    names  = OCTET_ORDER
    m_vals = [masses[n] for n in names]
    labels = [LATEX_LABELS[n] for n in names]
    colors = ["gold" if n == "chi0" else "steelblue" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), m_vals, color=colors,
                  edgecolor="black", linewidth=0.7)
    for i, (bar, n) in enumerate(zip(bars, names)):
        dm = splits[n]
        label = "DM" if n == "chi0" else f"+{dm:.1f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            label,
            ha="center",
            fontsize=8,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"$M\;[\mathrm{GeV}]$", fontsize=12)
    ax.set_title(r"Dark octet mass spectrum", fontsize=13)
    ax.set_ylim(150, 225)
    ax.yaxis.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def spectrum_csv_lines() -> list:
    masses = octet_spectrum()
    splits = mass_splittings(masses)
    lines  = ["state,I,Y,T,M_GeV,DeltaM_GeV"]
    for name in OCTET_ORDER:
        qn = OCTET_QN[name]
        m  = masses[name]
        dm = splits[name]
        lines.append(
            f"{name},{qn['I']},{qn['Y']},{qn['T']},{m:.4f},{dm:.4f}"
        )
    return lines
