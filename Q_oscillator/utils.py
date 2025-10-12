import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

root = Path(os.getcwd())
datapath = root / "Data"
plotpath = root / "plots"


def keysight_unc(v_div, dc_vo=0):
    """
    v_div: volts per division
    dc_vo: dc vertical offset
    """
    divs = 8  # hardset
    fs = divs * v_div  # full scale voltage range

    # voltage uncertainty
    dc_vga_unc = 0.03 if (v_div >= 0.01) else 0.04
    dc_vga = dc_vga_unc * fs  # dc vertical gain accuracy
    dc_voa = 0.1 * v_div + 0.002 + 0.01 * dc_vo  # dc vertical offset error
    qa = 1 / (2**8) * fs  # 8-bit quantization error 1/2^8 ~ 2%

    vu = dc_vga + dc_voa + qa  # worst-case voltage uncertainty

    return vu
    # time uncertainty


def meterman37xr_unc(reading: str, mode="V"):
    whole, dec = reading.split(".")
    sig = 10 ** (-len(dec))
    if mode == "V":
        return 0.001 * float(reading) + 5 * sig

    if mode == "R":
        return 0.005 * float(reading) + 8 * sig


print(meterman37xr_unc("520.1", "R"))


# Plot residuals function
def plot_residuals(data, y, yfit, sig, xlab="", name=""):
    residuals = y - yfit
    plt.errorbar(
        data, residuals, yerr=sig, fmt="o", color="k", alpha=0.8, label="Residuals"
    )
    plt.axhline(0, color="r", linestyle="--")
    # plt.fill_between(bin_centers, -sig, sig, color='gray', alpha=0.2, label='±1σ')
    plt.xlabel(rf"{xlab}")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()
    if name != "":
        plt.savefig(plotpath / name, bbox_inches="tight")


def format_parameters_with_errors(parameters, errors):
    def round_to_significant_figures(x, sig_figs=1):
        if x == 0:
            return 0
        else:
            return round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)

    formatted_values = []
    for param, err in zip(parameters, errors):
        # Round error to one significant figure
        rounded_err = round_to_significant_figures(err, sig_figs=1)
        # Adjust parameter value to match error precision
        param_precision = int(np.floor(np.log10(rounded_err)))
        rounded_param = round(param, -param_precision)

        formatted_values.append(f"{rounded_param} ± {rounded_err}")

    return formatted_values
