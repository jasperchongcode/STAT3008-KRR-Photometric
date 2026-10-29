import numpy as np
import numpy.typing as npt
import pandas as pd
from functools import partial
from pathlib import Path
from scipy.spatial.distance import pdist
from models.krr import KernelRidgeRegression
from models.matern import Matern

KERNEL_NAMES = {0.5: "Matern 1/2",
                1.5: "Matern 3/2",
                2.5: "Matern 5/2",
                np.inf: "Gaussian"}
NUS = list(KERNEL_NAMES)
LENGTH_FACTORS = [1/8, 1/4, 1/2, 1, 2, 4]
LAMBDAS = 10.0 ** np.arange(-8, 0)
BETAS = [0.25, 0.5, 1]
METHODS = [("P0", None), ("P1", None), ("P2", None)] + [("P3", beta) for beta in BETAS]

N_TEST = 10000
N_VALIDATION = 2000
N_TRAIN = 1000
SEEDS = range(5)
CROSSING = (0.33, 0.40)
PROPORTIONS = [1.0, 0.8, 0.6, 0.4]

BANDS = ["u", "g", "r", "i", "z"]

"""
Maps magnitudes x = (u, g, r, i, z) to one of the input representations.

P0 - Raw magnitudes x
P1 - Colours Dx = (u - g, g - r, r - i, i - z)
P2 - Projection Px, which subtracts the mean magnitude from every band
P3 - Colours plus the mean magnitude, (Dx, mean(x))

Inputs:
X - The magnitudes, of shape (n, 5)
method - One of "P0", "P1", "P2" or "P3"

Returns the features, of shape (n, d)
"""
def Preprocess(X: npt.NDArray[np.float64],
               method: str) -> npt.NDArray[np.float64]:

    colours = X[:, :-1] - X[:, 1:]
    mean_magnitude = np.mean(X, axis = 1, keepdims = True)

    if method == "P0":
        return X
    if method == "P1":
        return colours
    if method == "P2":
        return X - mean_magnitude
    if method == "P3":
        return np.hstack([colours, mean_magnitude])

    raise ValueError("method must be one of P0, P1, P2 or P3")

"""
Standardises each feature using the mean and standard deviation of the training set.

Inputs:
train - The training features, of shape (m, d)
X - The features to standardise, of shape (n, d)

Returns the standardised features, of shape (n, d)
"""
def Standardise(train: npt.NDArray[np.float64],
                X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

    return (X - np.mean(train, axis = 0)) / np.std(train, axis = 0)

"""
Keeps the fraction p of galaxies with the highest redshift, in a random order.

Inputs:
X - The features, of shape (n, d)
y - The redshifts, of shape (n)
p - The fraction of galaxies to keep
seed - The seed for the random order

Returns the kept features and redshifts
"""
def SelectTop(X: npt.NDArray[np.float64],
              y: npt.NDArray[np.float64],
              p: float = 1.0,
              seed: int = 0) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    cutoff = round(y.size * (1.0 - p))
    indices = np.random.default_rng(seed).permutation(np.argsort(y, kind = "stable")[cutoff:])

    return X[indices, :], y[indices]

"""
Draws disjoint training and validation indices from range(n).

Samplers:
permutation - Shuffles range(n); the first n_val indices are validation, the next n_train training
choice - Draws n_train + n_val indices without replacement; the first n_train are training, the rest validation

Inputs:
n - The number of points to draw from
n_train, n_val - The training and validation set sizes
seed - The seed
sampler - "permutation" or "choice"

Returns the training and validation indices
"""
def SplitIndices(n: int,
                 n_train: int,
                 n_val: int,
                 seed: int,
                 sampler: str = "permutation") -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:

    rng = np.random.default_rng(seed)

    if sampler == "permutation":
        indices = rng.permutation(n)
        return indices[n_val:n_val + n_train], indices[:n_val]
    if sampler == "choice":
        indices = rng.choice(n, n_train + n_val, replace = False)
        return indices[:n_train], indices[n_train:]

    raise ValueError("sampler must be 'permutation' or 'choice'")

"""
Calculates the evaluation metrics, using the normalised residual delta = (z_pred - z) / (1 + z).

Inputs:
z_pred - The predicted redshifts, of shape (n)
z_true - The true redshifts, of shape (n)

Returns a dictionary of metrics
"""
def Metrics(z_pred: npt.NDArray[np.float64],
            z_true: npt.NDArray[np.float64]) -> dict[str, float]:

    delta = (z_pred - z_true) / (1 + z_true)
    bias = np.median(delta)
    crossing = (z_true >= CROSSING[0]) & (z_true <= CROSSING[1])

    return {"rmse": np.sqrt(np.mean((z_pred - z_true) ** 2)),
            "bias": bias,
            "nmad": 1.48 * np.median(np.abs(delta - bias)),
            "eta_0.15": np.mean(np.abs(delta) > 0.15),
            "eta_0.05": np.mean(np.abs(delta) > 0.05),
            "rmse_crossing": np.sqrt(np.mean((z_pred[crossing] - z_true[crossing]) ** 2))}

"""
For each kernel, selects (length scale, lambda) by validation RMSE and evaluates the selected
model on the test set. Length scales are multiples of the median pairwise distance between
training points. The features should already be preprocessed and standardised.

Inputs:
X_train, y_train - The training features, of shape (m, d), and targets, of shape (m)
X_val, y_val - The validation features and targets
X_test, y_test - The test features and targets
labels - Values identifying the configuration, copied into every row
nus, length_factors, lambdas - The hyperparameter grid

Returns the validation rows (one per kernel, length scale and lambda) and the test rows
(one per kernel)
"""
def TuneAndEvaluate(X_train: npt.NDArray[np.float64],
                    y_train: npt.NDArray[np.float64],
                    X_val: npt.NDArray[np.float64],
                    y_val: npt.NDArray[np.float64],
                    X_test: npt.NDArray[np.float64],
                    y_test: npt.NDArray[np.float64],
                    labels: dict,
                    nus: list[float] = NUS,
                    length_factors: list[float] = LENGTH_FACTORS,
                    lambdas: npt.NDArray[np.float64] = LAMBDAS) -> tuple[list[dict], list[dict]]:

    validation_rows = []
    test_rows = []
    l_med = np.median(pdist(X_train))

    for nu in nus:
        best_rmse, best_model = np.inf, None

        for a in length_factors:
            kernel = partial(Matern, nu = nu, length_scale = a * l_med)
            K = kernel(X_train, X_train)

            for l in lambdas:
                model = KernelRidgeRegression()
                model.fit(X_train, y_train, l, kernel, K)
                rmse = np.sqrt(model.score(X_val, y_val))
                validation_rows.append({**labels, "nu": nu, "length_factor": a, "lambda": l, "rmse": rmse})

                if rmse < best_rmse:
                    best_rmse, best_model, best_a, best_l = rmse, model, a, l

        if best_model is None:
            raise RuntimeError(f"No valid model for nu = {nu} ({labels})")

        metrics = Metrics(best_model.predict(X_test), y_test)
        test_rows.append({**labels, "nu": nu, "length_factor": best_a, "lambda": best_l, **metrics})

    return validation_rows, test_rows

"""
Standardises the features with the training statistics, multiplies the last feature by beta
(the weight on the mean magnitude in P3) if given, then runs TuneAndEvaluate.

Inputs:
X_train, y_train, X_val, y_val, X_test, y_test - The preprocessed features and targets
labels - Values identifying the configuration, copied into every row
beta - The weight on the last feature, or None to leave it unscaled

Returns the validation rows and the test rows
"""
def StandardiseAndEvaluate(X_train: npt.NDArray[np.float64],
                           y_train: npt.NDArray[np.float64],
                           X_val: npt.NDArray[np.float64],
                           y_val: npt.NDArray[np.float64],
                           X_test: npt.NDArray[np.float64],
                           y_test: npt.NDArray[np.float64],
                           labels: dict,
                           beta: float | None = None) -> tuple[list[dict], list[dict]]:

    train, val, test = (Standardise(X_train, X) for X in (X_train, X_val, X_test))

    if beta is not None:
        for features in (train, val, test):
            features[:, -1] *= beta

    return TuneAndEvaluate(train, y_train, val, y_val, test, y_test, labels)

"""
Writes the validation and test rows to results_dir/<name>_validation.csv and <name>_test.csv,
and prints the mean and standard deviation of the test RMSE over seeds.

Inputs:
name - The experiment name, e.g. "real" or "simulated"
validation_rows, test_rows - The rows from the experiment
group_by - The columns identifying a configuration in the printed summary
results_dir - The directory to write the CSVs to
"""
def SaveResults(name: str,
                validation_rows: list[dict],
                test_rows: list[dict],
                group_by: list[str],
                results_dir: Path) -> None:

    results_dir.mkdir(parents = True, exist_ok = True)
    pd.DataFrame(validation_rows).to_csv(results_dir / f"{name}_validation.csv", index = False)
    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(results_dir / f"{name}_test.csv", index = False)

    print(test_results.groupby(group_by, dropna = False)["rmse"].agg(["mean", "std"]).to_string())
