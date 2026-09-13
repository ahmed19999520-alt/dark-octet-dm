from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass

from dark_octet.spectrum.gmo_formula import GMOMassOperator, DARK_OCTET_STATES
from dark_octet.constants import BenchmarkParameters, BENCH


@dataclass
class DarkOctetSummary:
    params: BenchmarkParameters = None

    def __post_init__(self):
        if self.params is None:
            self.params = BenchmarkParameters()
        self._gmo = GMOMassOperator(self.params)

    @property
    def masses(self) -> dict[str, float]:
        return self._gmo.all_masses()

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._gmo.mass_spectrum_dataframe()

    @property
    def ratios(self) -> dict:
        return self._gmo.mass_ratios()

    def stability_summary(self) -> dict[str, dict]:
        from dark_octet.boltzmann.decay_rates import compute_all_decay_rates

        rates = compute_all_decay_rates(
            masses_gev=self.masses,
            m_dm_gev=self.masses["chi_0"],
        )
        return rates

    def plot_weight_diagram(self, save_path: str = "weight_diagram.pdf") -> None:
        fig, ax = plt.subplots(figsize=(8, 7))

        positions = {
            "chi_pp":  ( 1.5,  1.2),
            "chi_p":   ( 0.5,  1.2),
            "chi_0":   (-0.5,  1.2),
            "chi_m":   (-1.5,  1.2),
            "Sigma_p": ( 1.0,  0.0),
            "Sigma_0": ( 0.0,  0.0),
            "Sigma_m": (-1.0,  0.0),
            "Lambda":  ( 0.2, -0.15),
        }

        labels = {
            "chi_pp":  r"$\chi^{++}$",
            "chi_p":   r"$\chi^{+}$",
            "chi_0":   r"$\chi^{0}$ (DM)",
            "chi_m":   r"$\chi^{-}$",
            "Sigma_p": r"$\Sigma^{+}_{\rm d}$",
            "Sigma_0": r"$\Sigma^{0}_{\rm d}$",
            "Sigma_m": r"$\Sigma^{-}_{\rm d}$",
            "Lambda":  r"$\Lambda_{\rm d}$",
        }

        masses = self.masses
        m_chi0 = masses["chi_0"]

        for name, (x, y) in positions.items():
            is_dm = name == "chi_0"
            circle = plt.Circle(
                (x, y),
                radius=0.12,
                color="gold" if is_dm else "#003087",
                alpha=0.9,
                zorder=5,
            )
            ax.add_patch(circle)

            label_text = labels[name]
            if masses[name] != m_chi0:
                delta_m = masses[name] - m_chi0
                label_text += f"\n$\\Delta M={delta_m:+.1f}$"

            ax.text(
                x,
                y + 0.18,
                label_text,
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

        chi0_pos = positions["chi_0"]
        for name, pos in positions.items():
            if name == "chi_0":
                continue
            if masses[name] > m_chi0 - 1.0:
                ax.annotate(
                    "",
                    xy=chi0_pos,
                    xytext=pos,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="gray",
                        lw=0.8,
                        linestyle="dashed",
                        connectionstyle="arc3,rad=0.1",
                    ),
                    zorder=3,
                )

        ax.axhline(y=1.2, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.axhline(y=0.0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.axvline(x=0.0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)

        ax.text(-2.2, 1.2, r"$Y_{\rm dark}=+1$", fontsize=9, color="gray", va="center")
        ax.text(-2.2, 0.0, r"$Y_{\rm dark}=0$", fontsize=9, color="gray", va="center")

        for x_val, label in [(-1.5, r"$-3/2$"), (-0.5, r"$-1/2$"), (0.5, r"$+1/2$"), (1.5, r"$+3/2$")]:
            ax.text(x_val, -0.55, label, ha="center", fontsize=9, color="gray")

        ax.set_xlabel(r"$I_{3,{\rm dark}}$", fontsize=13)
        ax.set_ylabel(r"$Y_{\rm dark}$", fontsize=13)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.7, 1.7)
        ax.set_title(
            r"Dark-Sector Baryon Octet"
            + "\n"
            + r"$\pi_2(\mathrm{SU}(3)/[\mathrm{SO}(3)\times\mathbb{Z}_2]) = \mathbb{Z}$",
            fontsize=12,
        )
        ax.set_aspect("equal")
        ax.grid(False)

        ax.annotate(
            r"$\to \chi^0 + \varphi$",
            xy=(0.5, 1.0),
            fontsize=8,
            color="gray",
            style="italic",
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_mass_spectrum_bar(self, save_path: str = "mass_spectrum.pdf") -> None:
        df = self.dataframe
        m_chi0 = df.loc[df["Name"] == "chi_0", "Mass_GeV"].values[0]

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = [
            "gold" if row["Is_DM"] else "#003087"
            for _, row in df.iterrows()
        ]
        labels = [row["State"] for _, row in df.iterrows()]
        masses = df["Mass_GeV"].values

        bars = ax.bar(range(len(masses)), masses, color=colors, edgecolor="white", linewidth=0.5)

        ax.axhline(y=m_chi0, color="gold", linewidth=1.5, linestyle="--", alpha=0.7, label=r"$m_{\chi^0}$")

        for i, (bar, mass) in enumerate(zip(bars, masses)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                mass + 1.0,
                f"{mass:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
            )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
        ax.set_ylabel("Mass [GeV]", fontsize=13)
        ax.set_ylim(120, 185)
        ax.set_title(
            f"Dark Octet Mass Spectrum  "
            f"[$M_0={self.params.M0}$, $\\beta={self.params.beta}$, "
            f"$\\gamma={self.params.gamma}$, $\\delta={self.params.delta}$, "
            f"$\\eta={self.params.eta}$] GeV",
            fontsize=12,
        )
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()