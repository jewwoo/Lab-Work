# Physics Lab-Work (Code + Reports)

This repo is a collection of my physics labwork: **analysis code + Jupyter notebooks + exported figures + written reports**. It currently includes two modules:

- **Currents in LCR** (RC / LR / LCR circuits) — uses an oscilloscope for time-domain transients and frequency-response measurements
- **Quantum Oscillator** — oscillatory motion / ringdown-style analysis, also using oscilloscope data where applicable

The goal is to keep everything **reproducible**: raw data is stored alongside the analysis notebooks/scripts that generate the plots and fitted parameters used in the reports.

---

## Repo Organization

Lab-Work/

├── Currents_in_LCR/

│ ├── Data/ # raw/processed circuit data (scope exports, etc.)

│ ├── e1_analysis.ipynb # Exercise 1 analysis

│ ├── e2_analysis.ipynb # Exercise 2 analysis

│ ├── e3_analysis.ipynb # Exercise 3 analysis

│ ├── utils.py # helper functions (fits, uncertainty, plotting, etc.)

│ └── Report_CurrentsinLCR.pdf # compiled report

│
└── Q_oscillator/

├── data/ # raw/processed oscillator data

├── Q_oscillator.ipynb # main analysis notebook

├── utils.py # helper functions for this module

└── *.png # exported figures (fits, residuals, etc.)



Notes:
- `__pycache__/` and `.DS_Store` appear from local runs / macOS and can be ignored (or added to `.gitignore`).

---

## How to Run

### 1) Create an environment (recommended)
If you have a `requirements.txt` or `environment.yml`, use it. Otherwise, the typical stack is:

```bash
pip install numpy scipy pandas matplotlib jupyter

From the repo root:
jupyter lab

Then open:

Currents_in_LCR/e1_analysis.ipynb, e2_analysis.ipynb, e3_analysis.ipynb
or
Q_oscillator/Q_oscillator.ipynb


**Currents in LCR**

Focuses on circuit dynamics and frequency response, typically including:

RC / LR transients (time constants, fits)

LCR resonance behavior (amplitude ratio vs frequency, Q-factor, bandwidth, etc.)

Comparisons between fitted parameters and component nominal/measured values

Residuals / goodness-of-fit checks (when applicable)

Files:

Notebooks: e1_analysis.ipynb, e2_analysis.ipynb, e3_analysis.ipynb

Report: Report_CurrentsinLCR.pdf

Helpers(uncertainty): utils.py

**Quantum Oscillator**

Focuses on oscillatory motion and damping / ringdown-type fits, typically including:

Exponential envelope fits

Extracting characteristic times / damping parameters

Comparing models and visualizing residuals

Files:

Notebook: Q_oscillator.ipynb

Data: data/

Figures: *.png

Helpers(uncertainty): utils.py

**Data & Instrument Notes (Oscilloscope)**

Both modules use oscilloscope measurements. Data is commonly exported as CSV and stored under each experiment’s data folder.
Where relevant, analysis assumes:

time axis and voltage scaling come from the scope export

any probe attenuation / coupling / sample rate should be reflected in the data or documented in the notebook
