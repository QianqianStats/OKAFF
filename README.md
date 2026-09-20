# OKAFF experiments

This repository contains simulation and real-data experiments for Adaptive online kernel changepoint detection. It compares four main methods:

- **OKAFF**: Online Kernel Adaptive Forgetting Factor
- **NEWMA**
- **MMDEW**
- **Online RFF MMD**

The fixed-threshold experiments also include **ScanB** and **OK-CUSUM**. Experiments report average run length (ARL) under no change and expected detection delay (EDD) after a change.

## Project structure

```text
OKAFF/
├── src/                         Shared detector and threshold implementations
├── adaptive-thresholds/
│   ├── ARL/{d1,d20}/            Adaptive-threshold ARL experiments
│   ├── EDD/{d1,d20}/            Adaptive-threshold EDD experiments
│   ├── compare_arl_edd.py       Four-method comparison tables
│   ├── run_adaptive_d1.sh
│   └── run_adaptive_d20.sh
├── fixed-thresholds/
│   ├── ARL/{d1,d20}/            Fixed-threshold ARL experiments
│   ├── EDD/{d1,d20}/            Fixed-threshold EDD experiments
│   ├── ARL-vs-EDD-plot/{d1,d20}/
│   ├── run_fixed_d1.sh
│   └── run_fixed_d20.sh
├── example/                     Minimal OKAFF ARL-to-EDD workflow
├── runtime/                     Six-method streaming-runtime benchmarks
├── number of RFF/{d1,d20}/      OKAFF experiments with different RFF numbers
├── real-data/                   TCPD datasets and Jupyter notebooks
├── boxplot/                     Empirical distribution figures
└── time-dependent-thresholds-sample-paths-plot/
```

The experiment scripts locate `src` relative to the repository, so the runners can be launched from the repository root.

## Environment setup with uv

Python 3.11 is recommended.

```bash
cd /OKAFF
uv venv --python 3.11
uv pip install numpy pandas scipy scikit-learn matplotlib tqdm numexpr onlinecp ipython jupyter "setuptools==81.0.0"
```

The shell runners use `.venv/bin/python` by default. A different interpreter can be supplied through `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python bash adaptive-thresholds/run_adaptive_d20.sh --n-runs 10
```

## Adaptive-threshold experiments

Table 1 and Table 6
Run d20:

```bash
bash adaptive-thresholds/run_adaptive_d20.sh --n-runs 500
```

Table 5
Run d1:

```bash
bash adaptive-thresholds/run_adaptive_d1.sh --n-runs 500
```

Each runner executes all ARL scripts first, all EDD scripts second, and then `compare_arl_edd.py`. The default is 500 runs when `--n-runs` is omitted.

Results are stored by dimension and run count:

```text
adaptive-thresholds/ARL/d20/results/
  OKAFF-adaptive-thresholds-ARL-d20-nrun500/
  NEWMA-adaptive-thresholds-ARL-d20-nrun500/
  MMDEW-adaptive-thresholds-ARL-d20-nrun500/
  online_rff_mmd-adaptive-thresholds-ARL-d20-nrun500/

adaptive-thresholds/EDD/d20/results/
  OKAFF-adaptive-thresholds-EDD-d20-nrun500/
  NEWMA-adaptive-thresholds-EDD-d20-nrun500/
  MMDEW-adaptive-thresholds-EDD-d20-nrun500/
  online_rff_mmd-adaptive-thresholds-EDD-d20-nrun500/
```

The d1 layout is the same under the corresponding `d1` folders. Comparison tables are written to:

```text
adaptive-thresholds/comparison_results/
```

The adaptive EDD experiments exclude the StandardNormal case and evaluate only post-change distributions.


## Fixed-threshold experiments

Figures 6,8,9,10
Run d20:

```bash
bash fixed-thresholds/run_fixed_d20.sh --n-runs 200
```

This runs:

1. d20 ARL experiments
2. d20 EDD experiments using the matching ARL CSVs
3. d20 full and short ARL-versus-EDD plots


Figures 7,11,12,13
Run d1:

```bash
bash fixed-thresholds/run_fixed_d1.sh --n-runs 200
```

This runs:

1. d1 ARL experiments
2. d1 EDD experiments using the matching ARL CSVs
3. d1 full and short ARL-versus-EDD plots

The default is 200 runs when `--n-runs` is omitted. CSV filenames include the selected run count, for example:

```text
fixed-thresholds/ARL/d20/results/OKAFF-ARL-d20-nrun200.csv
fixed-thresholds/EDD/d20/results/OKAFF-EDD-d20-nrun200.csv
fixed-thresholds/EDD/d20/results/OKAFF-ARL-EDD-d20-nrun200.csv
```

Plot outputs are written to:

```text
fixed-thresholds/ARL-vs-EDD-plot/d20/arl_edd_d20_full/
fixed-thresholds/ARL-vs-EDD-plot/d20/arl_edd_d20_short/
fixed-thresholds/ARL-vs-EDD-plot/d1/arl_edd_d1_full/
fixed-thresholds/ARL-vs-EDD-plot/d1/arl_edd_d1_short/
```

The fixed-threshold d1 experiments currently include OKAFF, NEWMA, Online RFF MMD, ScanB, and OK-CUSUM. There is no fixed-threshold d1 MMDEW experiment. 

Small Monte Carlo runs can produce unstable ARL estimates. If no estimate falls inside an EDD script's preferred ARL range, the script prints a warning and uses all valid thresholds from the matching ARL CSV.


## Real-data notebooks

The TCPD notebooks are under `real-data/`. The cleaned Oracle notebooks produce two tables:

- Overall best F1 results
- best F1 configuration per dataset

The OKAFF, NEWMA, MMDEW, and Online RFF MMD notebooks load detector implementations from `src`.

Tables 2,3,4
Launch Jupyter with:
```bash
cd OKAFF/real-data
../.venv/bin/jupyter lab
```


## Runtime benchmarks 
(Figures 14,15)

The runtime experiments compare NEWMA, OKAFF, Online RFF MMD, MMDEW, ScanB, and OK-CUSUM. Run them from the repository root:

```bash
.venv/bin/python runtime/benchmark_average_streaming_loglog_all_methods_d20.py
.venv/bin/python runtime/benchmark_average_streaming_loglog_all_methods_vs_dimension_T20000.py
```

The default is `--n-jobs 1`. Each worker runs all six methods sequentially for one repetition.

The first script measures streaming runtime at `d=20` for sequence lengths from 10 to 200,000. The second fixes the sequence length at `T=20,000` and uses dimensions from 1 to 5,000. Both use 10 repetitions and a 250-observation reference/burn-in. They print the runtime of every repetition and save average-runtime plots as PDF and PNG files:

```text
runtime/runtime_vs_length_gauss_d20/
  runtime_vs_length_all_six_methods_measurements_d20.csv
  runtime_vs_length_all_six_methods_mean_d20.csv
  runtime_vs_length_all_six_methods_loglog_d20.{pdf,png}

runtime/runtime_vs_dimension_gauss_T20000/
  runtime_vs_dimension_all_six_methods_measurements_T20000.csv
  runtime_vs_dimension_all_six_methods_mean_T20000.csv
  runtime_vs_dimension_all_six_methods_loglog_T20000.{pdf,png}
```

## Number of random Fourier features 
(Figures 16,17)

These experiments compare OKAFF with `m=500`, `m=1000`, and `m=5000` random Fourier features. Each runner computes ARL, computes EDD using the matching ARL results, and plots the three ARL–EDD curves.

Run the d1 study for the post-change distribution `N(-3, 1)`:

```bash
N_RUNS=200 PYTHON_BIN=.venv/bin/python \
  bash "number of RFF/d1/run_okaff_d1_m500_5000.sh"
```

Run the d20 study for the post-change distribution `N(0, 0.3I)`:

```bash
N_RUNS=200 PYTHON_BIN=.venv/bin/python \
  bash "number of RFF/d20/run_okaff_d20_m500_5000.sh"
```

Replace `N_RUNS=200` to change the Monte Carlo run count. The d1 runner also accepts `--full-grid`; its default uses thresholds 4.5, 7, and 9. The Python ARL and EDD scripts in both dimension folders accept `--features`, `--n-runs`, and `--full-grid` when run directly.

CSV results are saved under `number of RFF/d1/results/` or `number of RFF/d20/results/`. For every feature count, the files are:

```text
OKAFF-ARL-d{D}-m{M}-nrun{N}.csv
OKAFF-EDD-d{D}-m{M}-nrun{N}.csv
OKAFF-ARL-EDD-d{D}-m{M}-nrun{N}.csv
```

The comparison figures are saved as:

```text
number of RFF/d1/okaff_mean03_edd-d1.pdf
number of RFF/d20/okaff_cov03_edd-d20.pdf
```

## Reproducibility and runtime

Changing `--n-runs` changes Monte Carlo precision and therefore estimated ARL and EDD values, but it does not change detector settings.

Full runs can take a long time. Use a small value such as `--n-runs 2` or `--n-runs 10` to verify the workflow before starting the final experiment.




## Figures and Tables in paper

Table 1 and Table 6
Run d20:
```bash
bash adaptive-thresholds/run_adaptive_d20.sh --n-runs 500
```

Table 5
Run d1:
```bash
bash adaptive-thresholds/run_adaptive_d1.sh --n-runs 500
```

Figures 6,8,9,10
Run d20:
```bash
bash fixed-thresholds/run_fixed_d20.sh --n-runs 200
```

Figures 7,11,12,13
Run d1:
```bash
bash fixed-thresholds/run_fixed_d1.sh --n-runs 200
```

Tables 2,3,4
```bash
cd OKAFF/real-data
../.venv/bin/jupyter lab
```

Figures 14,15
```bash
.venv/bin/python runtime/benchmark_average_streaming_loglog_all_methods_d20.py
.venv/bin/python runtime/benchmark_average_streaming_loglog_all_methods_vs_dimension_T20000.py
```

Figures 16,17
```bash
N_RUNS=200 PYTHON_BIN=.venv/bin/python \
  bash "number of RFF/d1/run_okaff_d1_m500_5000.sh"
```
```bash
N_RUNS=200 PYTHON_BIN=.venv/bin/python \
  bash "number of RFF/d20/run_okaff_d20_m500_5000.sh"
```

Figures 3,4,5, 18-25
jupyter lab: ''boxplot'' is for all boxplots and related sample paths figures

Figures 1,2 
jupyter lab: ''time-dependent-thresholds-sample-paths-plot''









## OKAFF adaptive-threshold example

The `example/` folder provides a smaller end-to-end OKAFF workflow at `d=20`. Run it from the repository root:

```bash
bash example/run_okaff_adaptive_example.sh --n-runs 500
```

The runner executes `okaff_adaptive_threshold_arl.py` first, executes `okaff_adaptive_threshold_edd.py` second, and then joins their results by quantile. It prints the combined table and saves:

```text
example/results/OKAFF-adaptive-thresholds-ARL-EDD-d20-nrun500.csv
```

The combined CSV contains:

```text
prechange_distribution
postchange_distribution
q
ARL
ARL_se
ARL_censor_rate
EDD
EDD_se
EDD_censor_rate
```

The current example uses a standard Gaussian pre-change distribution and the post-change cases `N(0, 0.3I)` and `Uniform([-1,1]^20)`. `EDD_censor_rate` is taken from the EDD summary failure rate: runs that survive the pre-change period but do not signal by the maximum delay.

Set `--n-runs` to any positive integer. To use another Python interpreter, set `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python bash example/run_okaff_adaptive_example.sh --n-runs 10
```




## Acknowledgements

This repository includes or builds on code and data from the following projects:

- The implementation under `src/mmdew/` is adapted from [MMDEW](https://github.com/FlopsKa/mmdew-change-detector).
- The Online RFF MMD implementation is based on [rff-change-detection](https://github.com/FlopsKa/rff-change-detection).
- `real-data/datasets/`, `real-data/annotations.json`, and `real-data/metrics.py` are from [TCPDBench](https://github.com/alan-turing-institute/TCPDBench).


