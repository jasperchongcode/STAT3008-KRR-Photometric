import numpy as np
import numpy.typing as npt
from pathlib import Path
from models.simulated_data import GenerateSpectrum, Redshift, LoadFilterCurves, Photometry, AddBrightness
from util.evaluation import (Preprocess, SplitIndices, StandardiseAndEvaluate, SaveResults,
                             METHODS, N_TEST, N_VALIDATION, N_TRAIN, SEEDS)

N_POOL = 20000          # galaxies simulated per dataset
W_VALUES = [5, 25, 100, 400]

"""
Simulates the shape magnitudes of n galaxies, with B ~ U(1.1, 2.1) and z ~ U(0.01, 0.8).

Inputs:
n - The number of galaxies
w - The break width in Angstrom
filters - The response curves, as from LoadFilterCurves
rng - The random number generator

Returns the shape magnitudes, of shape (n, 5), and the redshifts, of shape (n)
"""
def SimulateShapes(n: int,
                   w: float,
                   filters: dict,
                   rng: np.random.Generator) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    z = rng.uniform(0.01, 0.8, size = n)
    B = rng.uniform(1.1, 2.1, size = n)
    magnitudes = np.empty((n, len(filters)))

    for k in range(n):
        spectrum = Redshift(GenerateSpectrum(4000.0, B[k], w), z[k])
        magnitudes[k] = Photometry(spectrum, filters)

    return magnitudes, z

"""
Runs the protocol for one dataset and input representation. The first N_TEST galaxies are the
fixed test set. For each seed, draws training and validation sets from the remaining galaxies,
then for each kernel selects (length scale, lambda) by validation RMSE and evaluates the selected
model on the test set. All configurations use the same splits for a given seed, so comparisons
are paired.

Inputs:
X - The magnitudes, of shape (n, 5)
z - The redshifts, of shape (n)
method - The input representation, one of "P0", "P1", "P2" or "P3"
beta - The weight on the mean magnitude, used only by P3 (None otherwise)
labels - Values identifying the configuration, copied into every row

Returns the validation rows (one per kernel, length scale and lambda) and the test rows
(one per kernel), for each seed
"""
def RunConfiguration(X: npt.NDArray[np.float64],
                     z: npt.NDArray[np.float64],
                     method: str,
                     beta: float | None,
                     labels: dict) -> tuple[list[dict], list[dict]]:

    labels = {**labels, "method": method, "beta": beta}
    features = Preprocess(X, method)
    test = np.arange(N_TEST)
    validation_rows = []
    test_rows = []

    for seed in SEEDS:
        train, validation = SplitIndices(len(z) - N_TEST, N_TRAIN, N_VALIDATION, seed, sampler = "permutation")
        train, validation = train + N_TEST, validation + N_TEST

        seed_validation, seed_test = StandardiseAndEvaluate(features[train], z[train],
                                                            features[validation], z[validation],
                                                            features[test], z[test],
                                                            {**labels, "seed": seed}, beta)
        validation_rows += seed_validation
        test_rows += seed_test

    return validation_rows, test_rows

"""
A1 (H1): break width, independent brightness, colour inputs.
A2 (H2): every input representation under both brightness settings, on the same shapes.

Inputs:
filter_path - The path to the SDSS filter response curves
results_dir - The directory to write simulated_validation.csv and simulated_test.csv to
"""
def RunSimulatedExperiment(filter_path: Path,
                           results_dir: Path) -> None:

    filters = LoadFilterCurves(filter_path)
    rng = np.random.default_rng(0)
    validation_rows = []
    test_rows = []

    for w in W_VALUES:
        print(f"A1: w = {w}")
        shapes, z = SimulateShapes(N_POOL, w, filters, rng)
        X = AddBrightness(shapes, z, "independent", rng)
        validation, test = RunConfiguration(X, z, "P1", None, {"experiment": "A1", "w": w, "brightness": "independent"})
        validation_rows += validation
        test_rows += test

    shapes, z = SimulateShapes(N_POOL, 25, filters, rng)
    for brightness in ["independent", "coupled"]:
        X = AddBrightness(shapes, z, brightness, rng)
        for method, beta in METHODS:
            print(f"A2: {brightness} brightness, {method}, beta = {beta}")
            validation, test = RunConfiguration(X, z, method, beta, {"experiment": "A2", "w": 25, "brightness": brightness})
            validation_rows += validation
            test_rows += test

    SaveResults("simulated", validation_rows, test_rows, ["experiment", "w", "brightness", "method", "beta", "nu"], results_dir)
