import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from util.evaluation import (Preprocess, SelectTop, SplitIndices, StandardiseAndEvaluate, SaveResults,
                             METHODS, PROPORTIONS, BANDS, N_TEST, N_VALIDATION, N_TRAIN, SEEDS)

"""
Runs the protocol on one dataset, in parallel over seeds. For each seed, draws training and
validation sets from the galaxies outside the test set, then for each kernel selects
(length scale, lambda) by validation RMSE and evaluates the selected model on the test set.

Inputs:
X_seen, y_seen - The features and redshifts that training and validation sets are drawn from
X_test, y_test - The fixed test set
method - The input representation
labels - Values identifying the configuration, copied into every row
beta - The weight on the mean magnitude, used only by P3

Returns the validation rows and the test rows, in seed order
"""
def TestConfiguration(X_seen: npt.NDArray[np.float64],
                      y_seen: npt.NDArray[np.float64],
                      X_test: npt.NDArray[np.float64],
                      y_test: npt.NDArray[np.float64],
                      method: str,
                      labels: dict,
                      beta: float | None = None) -> tuple[list[dict], list[dict]]:

    labels = {**labels, "method": method}
    jobs = []
    for seed in SEEDS:
        train, val = SplitIndices(len(y_seen), N_TRAIN, N_VALIDATION, seed, sampler = "choice")
        jobs.append(delayed(StandardiseAndEvaluate)(X_seen[train], y_seen[train], X_seen[val], y_seen[val],
                                                    X_test, y_test, {**labels, "seed": seed}, beta))

    validation_rows = []
    test_rows = []
    for validation, test in tqdm(Parallel(n_jobs = len(jobs), return_as = "generator")(jobs), total = len(jobs)):
        validation_rows += validation
        test_rows += test

    return validation_rows, test_rows

"""
Keeps the fraction p of galaxies with the highest redshift, splits off the first N_TEST as the
test set, and runs TestConfiguration on the rest.

Inputs:
X - The magnitudes, of shape (n, 5)
y - The redshifts, of shape (n)
method - The input representation
p - The fraction of galaxies to keep
labels - Values identifying the configuration, copied into every row
beta - The weight on the mean magnitude, used only by P3

Returns the validation rows and the test rows
"""
def RunOnTop(X: npt.NDArray[np.float64],
             y: npt.NDArray[np.float64],
             method: str,
             p: float,
             labels: dict,
             beta: float | None = None) -> tuple[list[dict], list[dict]]:

    X_top, y_top = SelectTop(Preprocess(X, method), y, p)
    return TestConfiguration(X_top[N_TEST:], y_top[N_TEST:], X_top[:N_TEST], y_top[:N_TEST], method, labels, beta)

"""
R1 (H1): colour inputs on the highest-redshift fraction p of galaxies.
R2 (H2): every input representation on the full sample.

Inputs:
data_path - The path to the SDSS data
results_dir - The directory to write real_validation.csv and real_test.csv to
"""
def RunRealExperiment(data_path: Path,
                      results_dir: Path) -> None:

    data = pd.read_csv(data_path)
    X = data[BANDS].to_numpy()
    y = data["redshift"].to_numpy()

    configurations = [("P1", None, p, {"p": p, "experiment": "R1"}) for p in PROPORTIONS]
    configurations += [(method, beta, 1.0, {"p": 1.0, "experiment": "R2", "beta": beta}) for method, beta in METHODS]

    validation_rows = []
    test_rows = []
    for method, beta, p, labels in configurations:
        print(f"{labels['experiment']}: method = {method}, beta = {beta}, p = {p}")
        validation, test = RunOnTop(X, y, method, p, labels, beta)
        validation_rows += validation
        test_rows += test

    SaveResults("real", validation_rows, test_rows, ["experiment", "p", "method", "beta", "nu"], results_dir)
