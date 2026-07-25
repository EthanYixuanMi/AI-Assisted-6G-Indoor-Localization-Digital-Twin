"""Shared, accessible visual language for paper and dashboard graphics."""

from __future__ import annotations

from collections.abc import Iterable
import re

import matplotlib

# The backend must be selected before pyplot is imported by any exporter.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

NEUTRAL = {
    "ink": "#222222",
    "mid": "#666666",
    "light": "#A6A6A6",
    "grid": "#D9D9D9",
    "paper": "#F7F8FA",
}

DARK = {
    "background": "#07111F",
    "panel": "#0D1B2A",
    "panel_alt": "#10263B",
    "grid": "#27445D",
    "text": "#E8F1F8",
    "muted": "#91A9BC",
    "cyan": "#56B4E9",
}

ALGORITHM_ORDER = [
    "Geometric LS",
    "KNN",
    "Direct AI",
    "Residual AI",
    "Geometric LS + Kalman",
    "KNN + Kalman",
    "Direct AI + Kalman",
    "Residual AI + Kalman",
]

ALGORITHM_COLORS = {
    "Geometric LS": NEUTRAL["mid"],
    "KNN": OKABE_ITO["sky_blue"],
    "Direct AI": OKABE_ITO["orange"],
    "Residual AI": OKABE_ITO["bluish_green"],
    "Geometric LS + Kalman": "#999999",
    "KNN + Kalman": OKABE_ITO["blue"],
    "Direct AI + Kalman": OKABE_ITO["vermillion"],
    "Residual AI + Kalman": OKABE_ITO["reddish_purple"],
}

ALGORITHM_LINESTYLES = {
    "Geometric LS": "--",
    "KNN": "-.",
    "Direct AI": ":",
    "Residual AI": "-",
    "Geometric LS + Kalman": "--",
    "KNN + Kalman": "-.",
    "Direct AI + Kalman": ":",
    "Residual AI + Kalman": "-",
}

ALGORITHM_MARKERS = {
    "Geometric LS": "o",
    "KNN": "s",
    "Direct AI": "^",
    "Residual AI": "D",
    "Geometric LS + Kalman": "o",
    "KNN + Kalman": "s",
    "Direct AI + Kalman": "^",
    "Residual AI + Kalman": "D",
}

SCENARIO_ORDER = [
    "normal",
    "high_noise",
    "strong_blockage",
    "anchor_failure",
    "domain_shift",
]

SCENARIO_LABELS = {
    "normal": "Normal",
    "high_noise": "High Noise",
    "strong_blockage": "Strong Blockage",
    "anchor_failure": "Anchor Failure",
    "domain_shift": "Domain Shift",
}


def _key(value: object) -> str:
    """Return an identifier-like normalization used for aliases."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


_ALGORITHM_ALIASES = {
    "geometric": "Geometric LS",
    "geometric_ls": "Geometric LS",
    "least_squares": "Geometric LS",
    "wls": "Geometric LS",
    "knn": "KNN",
    "knn_fingerprinting": "KNN",
    "fingerprinting": "KNN",
    "direct_ai": "Direct AI",
    "direct_ai_regression": "Direct AI",
    "mlp": "Direct AI",
    "mlp_regressor": "Direct AI",
    "residual_ai": "Residual AI",
    "residual_ai_correction": "Residual AI",
    "residual": "Residual AI",
    "residual_rf": "Residual AI",
    "geometric_ls_kalman": "Geometric LS + Kalman",
    "geometric_ls_filtered": "Geometric LS + Kalman",
    "knn_kalman": "KNN + Kalman",
    "knn_filtered": "KNN + Kalman",
    "direct_ai_kalman": "Direct AI + Kalman",
    "direct_ai_filtered": "Direct AI + Kalman",
    "residual_ai_kalman": "Residual AI + Kalman",
    "residual_ai_filtered": "Residual AI + Kalman",
}

_SCENARIO_ALIASES = {
    "normal": "normal",
    "baseline": "normal",
    "in_domain": "normal",
    "indomain": "normal",
    "high_noise": "high_noise",
    "noise": "high_noise",
    "strong_blockage": "strong_blockage",
    "blockage": "strong_blockage",
    "nlos": "strong_blockage",
    "anchor_failure": "anchor_failure",
    "anchor_dropout": "anchor_failure",
    "missing_anchor": "anchor_failure",
    "domain_shift": "domain_shift",
    "shift": "domain_shift",
}


def canonical_algorithm(value: object) -> str:
    """Map common model identifiers to stable report-facing names."""

    text = str(value).strip()
    return _ALGORITHM_ALIASES.get(_key(text), text)


def canonical_scenario(value: object) -> str:
    """Map common scenario labels to stable machine-facing identifiers."""

    key = _key(value)
    return _SCENARIO_ALIASES.get(key, key)


def ordered_algorithms(values: Iterable[object]) -> list[str]:
    """Return unique canonical algorithm names in a stable semantic order."""

    unique = list(dict.fromkeys(canonical_algorithm(value) for value in values))
    ranked = [name for name in ALGORITHM_ORDER if name in unique]
    ranked.extend(sorted(name for name in unique if name not in ranked))
    return ranked


def algorithm_color(name: object) -> str:
    """Return a stable accessible color, including for an unknown method."""

    canonical = canonical_algorithm(name)
    if canonical in ALGORITHM_COLORS:
        return ALGORITHM_COLORS[canonical]
    fallback = [
        OKABE_ITO["blue"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["yellow"],
        NEUTRAL["ink"],
    ]
    return fallback[sum(ord(char) for char in canonical) % len(fallback)]


def configure_paper_style() -> None:
    """Configure compact IEEE-friendly Matplotlib defaults."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "figure.titlesize": 10.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.2,
            "axes.grid": True,
            "grid.color": NEUTRAL["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def configure_dashboard_style() -> None:
    """Configure Matplotlib for the static dark dashboard composite."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.facecolor": DARK["background"],
            "axes.facecolor": DARK["panel"],
            "savefig.facecolor": DARK["background"],
            "text.color": DARK["text"],
            "axes.labelcolor": DARK["text"],
            "axes.titlecolor": DARK["text"],
            "axes.edgecolor": DARK["grid"],
            "xtick.color": DARK["muted"],
            "ytick.color": DARK["muted"],
            "grid.color": DARK["grid"],
            "grid.alpha": 0.55,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
