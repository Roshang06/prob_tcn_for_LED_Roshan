"""
Plots EVM% vs. quantization bit width for the SystemVerilog model,
comparing Post-Training Quantization (PTQ) against Quantization-Aware
Training (QAT).

Edit the EXAMPLE DATA section below with your own bit widths and EVM
arrays, then run:

    python plot_evm_vs_bitwidth.py
"""

import matplotlib.pyplot as plt
import numpy as np

# ------------------------- EXAMPLE DATA -------------------------
# Bit widths tested (x-axis). Same widths used for both PTQ and QAT.
bit_widths = [2, 4, 6, 8, 10, 12, 14, 16]

# EVM% for each bit width, Post-Training Quantization
ptq_evm = [100, 92.93, 49.50, 30.32, 20.88, 13.93, 12.15, 11.87]

# EVM% for each bit width, Quantization-Aware Training
qat_evm = [91.81, 58.77, 29.36, 21.09, 16.15, 13.65, 12.68, 12.27]
# ------------------------------------------------------------------


def plot_evm_vs_bitwidth(bit_widths, ptq_evm, qat_evm, out_path="evm_vs_bitwidth.png"):
    bit_widths = np.asarray(bit_widths)
    ptq_evm = np.asarray(ptq_evm)
    qat_evm = np.asarray(qat_evm)

    if not (len(bit_widths) == len(ptq_evm) == len(qat_evm)):
        raise ValueError("bit_widths, ptq_evm, and qat_evm must all be the same length")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.invert_xaxis()

    ptq_color = "#E4572E"
    qat_color = "#2E86AB"

    ax.plot(bit_widths, ptq_evm, marker="o", markersize=8, linewidth=2.5,
            color=ptq_color, label="Post-Training Quantization (PTQ)",
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    ax.plot(bit_widths, qat_evm, marker="s", markersize=8, linewidth=2.5,
            color=qat_color, label="Quantization-Aware Training (QAT)",
            markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    # Light fill under each curve for visual weight
    ax.fill_between(bit_widths, ptq_evm, color=ptq_color, alpha=0.08, zorder=1)
    ax.fill_between(bit_widths, qat_evm, color=qat_color, alpha=0.08, zorder=1)

    # Annotate each point with its EVM value
    for x, y in zip(bit_widths, ptq_evm):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color=ptq_color)
    for x, y in zip(bit_widths, qat_evm):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8.5, color=qat_color)

    ax.set_xlabel("Bit Width", fontsize=12, fontweight="bold")
    ax.set_ylabel("EVM (%)", fontsize=12, fontweight="bold")
    ax.set_title("EVM vs. Bit Width for Linear Quantization",
                 fontsize=14, fontweight="bold", pad=15)

    ax.set_xticks(bit_widths)
    ax.margins(y=0.15)
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, loc="upper right")
    ax.tick_params(labelsize=10)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved plot to {out_path}")

    return fig, ax


if __name__ == "__main__":
    plot_evm_vs_bitwidth(bit_widths, ptq_evm, qat_evm)