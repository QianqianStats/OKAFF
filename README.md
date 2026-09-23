# OKAFF experiments

This repository contains simulation and real-data experiments for Adaptive online kernel changepoint detection (http://arxiv.org/abs/2609.22545). It compares four main methods:

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

Python 3.11 is recommended. Run the following from the repository root with `uv` and Git installed:

```bash
cd /OKAFF
uv venv --python 3.11
uv pip install numpy pandas scipy scikit-learn matplotlib tqdm numexpr ipython jupyter "setuptools==81.0.0"
uv pip install "git+https://github.com/lightonai/newma.git"
```

Verify the required `onlinecp` imports:

```bash
.venv/bin/python -c "from onlinecp.algos import NEWMA; import onlinecp.utils.feature_functions; print('onlinecp installed successfully')"
```

The shell runners use `.venv/bin/python` by default. A different interpreter can be supplied through `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python bash adaptive-thresholds/run_adaptive_d20.sh --n-runs 10
```


## Adaptive-threshold experiments

(Table 1 and Table 6)
Run d20:

```bash
bash adaptive-thresholds/run_adaptive_d20.sh --n-runs 500
```

(Table 5)
Run d1:

```bash
bash adaptive-thresholds/run_adaptive_d1.sh --n-runs 500
```

Each runner executes all ARL scripts first, all EDD scripts second, and then `compare_arl_edd.py`. The default is 500 runs when `--n-runs` is omitted.


## Fixed-threshold experiments

(Figures 6,8,9,10)
Run d20:

```bash
bash fixed-thresholds/run_fixed_d20.sh --n-runs 200
```

This runs:

1. d20 ARL experiments
2. d20 EDD experiments 
3. d20 full and short ARL-versus-EDD plots


(Figures 7,11,12,13)
Run d1:

```bash
bash fixed-thresholds/run_fixed_d1.sh --n-runs 200
```

This runs:

1. d1 ARL experiments
2. d1 EDD experiments 
3. d1 full and short ARL-versus-EDD plots

The default is 200 runs when `--n-runs` is omitted. 

The fixed-threshold d1 experiments currently include OKAFF, NEWMA, Online RFF MMD, ScanB, and OK-CUSUM. There is no fixed-threshold d1 MMDEW experiment. 

Small Monte Carlo runs can produce unstable ARL estimates. If no estimate falls inside an EDD script's preferred ARL range, the script prints a warning and uses all valid thresholds from the matching ARL CSV.


## Real-data notebooks

The TCPD notebooks are under `real-data/`. The notebooks produce two tables:

- Overall best F1 results
- best F1 configuration per dataset

The OKAFF, NEWMA, MMDEW, and Online RFF MMD notebooks load detector implementations from `src`.

(Tables 2,3,4)
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

Replace `N_RUNS=200` to change the Monte Carlo run count. The d1 runner also accepts `--full-grid`; its default uses three thresholds. The Python ARL and EDD scripts in both dimension folders accept `--features`, `--n-runs`, and `--full-grid` when run directly.


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

Figures 3,4,5, 18-25:
jupyter lab: ''boxplot'' is for all boxplots and related sample paths figures

Figures 1,2:
jupyter lab: ''time-dependent-thresholds-sample-paths-plot''









## OKAFF adaptive-threshold example

The `example/` folder provides a smaller end-to-end OKAFF workflow at `d=20`. Run it from the repository root:

```bash
bash example/run_okaff_adaptive_example.sh --n-runs 500
```

The runner executes `okaff_adaptive_threshold_arl.py` first, executes `okaff_adaptive_threshold_edd.py` second, and then joins their results by quantile. 


Set `--n-runs` to any positive integer. To use another Python interpreter, set `PYTHON_BIN`:

```bash
PYTHON_BIN=/path/to/python bash example/run_okaff_adaptive_example.sh --n-runs 10
```




## Acknowledgements

This repository includes or builds on code and data from the following projects:

- The implementation under `src/mmdew/` is from [MMDEW](https://github.com/FlopsKa/mmdew-change-detector) and [rff-change-detection](https://github.com/FlopsKa/rff-change-detection). 
- `real-data/datasets/`, `real-data/annotations.json`, and `real-data/metrics.py` are from [TCPDBench](https://github.com/alan-turing-institute/TCPDBench).


