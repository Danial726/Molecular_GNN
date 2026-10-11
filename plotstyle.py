import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = ["#2f6f8f", "#c25b3f", "#4a9d6e", "#9a6fb0", "#d4a03a", "#7a7a7a"]
SHIFT_COLOR = {"scaffold": "#2f6f8f", "size": "#c25b3f", "label": "#4a9d6e"}


def apply():
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


def save(fig, name):
    import os
    os.makedirs("figures", exist_ok=True)
    fig.savefig(f"figures/{name}.png", bbox_inches="tight")
    fig.savefig(f"figures/{name}.svg", bbox_inches="tight")
    plt.close(fig)
