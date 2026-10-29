from util.dataloader import DownloadData
from experiments.run_real_experiment import RunRealExperiment
from experiments.run_simulated_experiment import RunSimulatedExperiment
from util.summarise import RealTables, SimulatedTables, PrintSelectionCounts
from pathlib import Path

RESULTS_DIR = Path("results")
DATA_PATH = Path("data/sdss_galaxies_d4000.csv")
FILTER_PATH = Path("data/sdss_filter_curves.fits")

# Load data
if not DATA_PATH.exists():
    DownloadData(DATA_PATH)

# Run experiments
RunSimulatedExperiment(FILTER_PATH, RESULTS_DIR)
RunRealExperiment(DATA_PATH, RESULTS_DIR)

# Summarise results
SimulatedTables(RESULTS_DIR / "simulated_test.csv", RESULTS_DIR)
RealTables(RESULTS_DIR / "real_test.csv", RESULTS_DIR, DATA_PATH)

# Summarise vals
PrintSelectionCounts("simulated", RESULTS_DIR / "simulated_test.csv")
PrintSelectionCounts("real", RESULTS_DIR / "real_test.csv")