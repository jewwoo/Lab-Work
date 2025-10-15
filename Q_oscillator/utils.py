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
    
    #Voltage Uncertainty
    if mode == "V":
        return 0.001 * float(reading) + 5 * sig
    
    #Resistance Uncertainty

    if mode == "R":
        return 0.005 * float(reading) + 4 * sig
    
    #Inductance Uncertainty
    
    if mode == "I":
        return 0.05 * float(reading) + 30 * sig
    
    
    #Capacitance Uncertainty
    
    if mode == "C":
        return 0.03 * float(reading) + 5 * sig


print(meterman37xr_unc("520.1", "R"))
import numpy as np


def freq_uncertainty(f_reading, lsb, ppm=50):
    """
    Calculate combined and relative uncertainties of frequency measurements
    from a Keysight 1000 X-Series oscilloscope.

    Parameters
    ----------
    f_reading : float or array-like
        Frequency reading(s) from the oscilloscope (in Hz).
    lsb : float or array-like
        Least significant digit(s) of each reading (in Hz).
    ppm : float, optional
        Timebase accuracy in parts per million (default = 50 ppm).

    Returns
    -------
    u_combined : ndarray
        Combined uncertainty in Hz.
    rel_uncertainty : ndarray
        Relative uncertainty in percent.
    """

    f = np.array(f_reading, dtype=float)
    lsb = np.array(lsb, dtype=float)

    u_tb = f * (ppm * 1e-6)       # timebase term
    u_disp = 0.5 * lsb            # display term
    u_combined = np.sqrt(u_tb**2 + u_disp**2)
    rel_uncertainty = (u_combined / f) * 100

    return u_combined, rel_uncertainty


# Example usage
frequencies = [10000, 5000, 15000]  # in Hz
lsbs = [1, 1, 1]                    # each with 1 Hz resolution
u, rel = freq_uncertainty(frequencies, lsbs)

for f, u_i, r_i in zip(frequencies, u, rel):
    print(f"{f/1000:.3f} kHz ± {u_i:.2f} Hz ({r_i:.5f}%)")



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


import numpy as np

def amplitude_ratio_uncertainty(Vout, Vin, u_Vout, u_Vin, return_db=False):
    """
    Uncertainty of amplitude ratio R = Vout / Vin (assumes independence).
    If return_db=True, also returns R_dB and its uncertainty.
    """
    Vout, Vin = np.asarray(Vout, float), np.asarray(Vin, float)
    u_Vout, u_Vin = np.asarray(u_Vout, float), np.asarray(u_Vin, float)

    R = Vout / Vin
    u_R = R * np.sqrt( (u_Vout/np.maximum(Vout, 1e-300))**2 +
                       (u_Vin /np.maximum(Vin , 1e-300))**2 )

    if not return_db:
        return R, u_R

    R_dB = 20*np.log10(np.maximum(R, 1e-300))
    u_R_dB = (20/np.log(10)) * (u_R / np.maximum(R, 1e-300))
    return R, u_R, R_dB, u_R_dB


def combine_x_uncert_into_y(f, u_f, model, params, u_y_from_volt):
    """
    Map frequency uncertainty into vertical uncertainty via model slope and
    add in quadrature with the measurement (voltage-based) vertical uncertainty.

    Parameters
    ----------
    f : array-like (Hz)
    u_f : array-like (Hz)  - from your freq_uncertainty() function
    model : callable       - M(f, *params) returning the modeled amplitude ratio
    params : tuple/list    - parameters for the model
    u_y_from_volt : array-like - vertical uncertainty from voltages (u_R)

    Returns
    -------
    u_y_total : ndarray
        sqrt( u_R^2 + (dM/df * u_f)^2 )
    """

    f = np.asarray(f, float)
    u_f = np.asarray(u_f, float)
    u_y_from_volt = np.asarray(u_y_from_volt, float)

    # numerical derivative dM/df using a small relative step
    eps = 1e-6
    df = np.maximum(eps * np.maximum(np.abs(f), 1.0), 1.0)  # at least 1 Hz step
    M_plus  = model(f + df, *params)
    M_minus = model(f - df, *params)
    dM_df = (M_plus - M_minus) / (2*df)

    u_y_from_x = np.abs(dM_df) * u_f
    u_y_total = np.sqrt(u_y_from_volt**2 + u_y_from_x**2)
    return u_y_total
