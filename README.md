# Kernel Ridge Regression for Photometric Redshift Estimation

## Authors

- Jasper Chong (48834207)
- Bailey Clarke (48023850)
- Adrian Durana (47995156)
- Sarah Elford (47433984)

## Usage

Run `main.py` from inside this folder. It writes all experimental results to `./results`:

```bash
python main.py
```

To generate figures, run `figures.ipynb`.

## Structure

- `experiments/`: the simulated (A1, A2) and real-data (R1, R2) experiments
- `models/`: kernel ridge regression, the Matérn kernels and the simulated data model
- `util/`: data download, shared evaluation code and results summaries

## Requirements

numpy, scipy, pandas, joblib, tqdm, astropy, requests and matplotlib