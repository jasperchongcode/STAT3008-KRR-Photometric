import numpy as np
import pandas as pd
from fractions import Fraction
from pathlib import Path
from util.evaluation import SelectTop, KERNEL_NAMES, LENGTH_FACTORS, LAMBDAS, CROSSING, PROPORTIONS, N_TEST

LATEX_KERNEL_NAMES = {nu: name.replace("Matern ", "Mat\\'ern $") + "$" if name.startswith("Matern") else name
                      for nu, name in KERNEL_NAMES.items()}

"""
Formats a LaTeX tabular with a left-aligned first column and centred other columns.

Inputs:
header - The column headings
rows - The cells of each row, as strings

Returns the LaTeX tabular
"""
def Tabular(header: list[str],
            rows: list[list[str]]) -> str:

    lines = [r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}",
             r"\toprule",
             " & ".join(header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]

    return "\n".join(lines)

"""
Adds a "kernel" column (the kernel's name) and a "representation" column (the method, with beta
for P3) to the results.

Inputs:
results - The test results
latex - Whether to use LaTeX names, for the .tex tables

Returns the results with the two columns added
"""
def AddLabels(results: pd.DataFrame,
              latex: bool = True) -> pd.DataFrame:

    names = LATEX_KERNEL_NAMES if latex else KERNEL_NAMES
    p3 = "P3 ($\\beta = {:g}$)" if latex else "P3 (beta = {:g})"

    return results.assign(kernel = results["nu"].map(names),
                          representation = [p3.format(beta) if method == "P3" else method
                                            for method, beta in zip(results["method"], results["beta"])])

"""
Writes each table to out_dir/<prefix>_<name>.tex.
"""
def WriteTables(tables: dict[str, str],
                prefix: str,
                out_dir: Path) -> None:

    out_dir.mkdir(parents = True, exist_ok = True)
    for name, table in tables.items():
        (out_dir / f"{prefix}_{name}.tex").write_text(table + "\n")

"""
Summarises one metric over seeds, with one row per value of rows and one column per kernel.
Each cell is the mean +/- standard deviation (scaled by 1000), with the smallest mean in each row in bold.

Inputs:
results - The test results, with labels from AddLabels
rows - The column of results that defines the table rows
metric - The metric to summarise, e.g. "rmse"
row_header - The heading of the first column

Returns the LaTeX tabular
"""
def SummaryTable(results: pd.DataFrame,
                 rows: str,
                 metric: str,
                 row_header: str) -> str:

    kernels = list(LATEX_KERNEL_NAMES.values())
    grouped = results.groupby([rows, "kernel"], sort = False)[metric]
    mean = grouped.mean().unstack("kernel")[kernels]
    std = grouped.std().unstack("kernel")[kernels]

    table = []
    for row in mean.index:
        cells = [str(row)]
        for kernel in kernels:
            cell = f"{1000 * mean.loc[row, kernel]:.2f} $\\pm$ {1000 * std.loc[row, kernel]:.2f}"
            if mean.loc[row, kernel] == mean.loc[row].min():
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        table.append(cells)

    return Tabular([row_header] + kernels, table)

"""
Formats a length factor a (length scale = a * l_med) as a fraction, e.g. 0.125 -> 1/8.
"""
def FormatLength(a: float) -> str:

    return str(Fraction(a).limit_denominator(64))

"""
Formats lambda as a power of ten, e.g. 1e-4 -> $10^{-4}$.
"""
def FormatLambda(l: float) -> str:

    return f"$10^{{{round(np.log10(l))}}}$"

"""
Formats, for each proportion p, how many galaxies the R1 dataset keeps and what share of them
lie in the crossing window. Uses SelectTop and N_TEST, so the test-set counts match the galaxies
rmse_crossing is computed on.

Inputs:
data_path - The path to the SDSS data
proportions - The values of p

Returns the LaTeX tabular
"""
def CrossingTable(data_path: Path,
                  proportions: list[float] = PROPORTIONS) -> str:

    z = pd.read_csv(data_path)["redshift"].to_numpy()
    table = []
    for p in proportions:
        _, z_top = SelectTop(z[:, None], z, p)
        in_window = (z_top >= CROSSING[0]) & (z_top <= CROSSING[1])
        table.append([str(p), f"{len(z_top):,}", f"{z_top.min():.3f}", f"{100 * in_window.mean():.2f}",
                      f"{in_window[:N_TEST].sum():,} of {min(N_TEST, len(z_top)):,}"])

    return Tabular(["$p$", "Galaxies", "Lowest $z$", r"In window (\%)", "Test galaxies in window"], table)

"""
Reads the simulated test results and writes the summary tables for the results section to
out_dir/simulated_<name>.tex.

Inputs:
path - The path to the test results
out_dir - The directory to write the tables to

Returns a dictionary mapping each table name to its LaTeX tabular
"""
def SimulatedTables(path: Path,
                    out_dir: Path) -> dict[str, str]:

    results = AddLabels(pd.read_csv(path))
    a1 = results[results["experiment"] == "A1"]
    a2 = results[results["experiment"] == "A2"]

    tables = {"a1_rmse": SummaryTable(a1, "w", "rmse", r"$w$ (\AA)"),
              "a1_crossing": SummaryTable(a1, "w", "rmse_crossing", r"$w$ (\AA)")}
    for brightness in ["independent", "coupled"]:
        tables[f"a2_{brightness}"] = SummaryTable(a2[a2["brightness"] == brightness], "representation", "rmse", "Input")

    WriteTables(tables, "simulated", out_dir)
    return tables

"""
Reads the real-data test results and writes the summary tables for the results section to
out_dir/real_<name>.tex.

r1_rmse - R1, RMSE with one row per proportion p of highest-redshift galaxies kept
r1_crossing - R1, RMSE in the crossing window, one row per p
r2_rmse - R2, RMSE with one row per input representation (P0, P1, P2, P3 for each beta)
r1_window - R1, the share of each dataset in the crossing window (see CrossingTable)
r1_length, r1_lambda - R1, the length factor and lambda selected by validation (see SelectionTable)

Inputs:
path - The path to the test results
out_dir - The directory to write the tables to
data_path - The path to the SDSS data, for r1_window

Returns a dictionary mapping each table name to its LaTeX tabular
"""
def RealTables(path: Path,
               out_dir: Path,
               data_path: Path) -> dict[str, str]:

    results = AddLabels(pd.read_csv(path))
    r1 = results[results["experiment"] == "R1"]
    r2 = results[results["experiment"] == "R2"]

    tables = {"r1_rmse": SummaryTable(r1, "p", "rmse", "$p$"),
              "r1_crossing": SummaryTable(r1, "p", "rmse_crossing", "$p$"),
              "r2_rmse": SummaryTable(r2, "representation", "rmse", "Input"),
              "r1_window": CrossingTable(data_path)}

    WriteTables(tables, "real", out_dir)
    return tables

"""
Counts how often one hyperparameter value was selected by validation, with one row per value
of rows and one column per kernel, plus a Total column (over kernels) and a Total row (over rows).
Each cell is "n_chosen / n_runs", where a run is one seed.

Inputs:
results - The test results for one experiment, with labels from AddLabels(latex = False)
rows - The column of results that defines the table rows
parameter - The selected hyperparameter, "length_factor" or "lambda"
value - The value of the hyperparameter to count

Returns the table as a DataFrame of strings
"""
def CountTable(results: pd.DataFrame,
               rows: str,
               parameter: str,
               value: float) -> pd.DataFrame:

    results = results.assign(chosen = np.isclose(results[parameter], value))
    row_order = list(results[rows].unique())
    kernels = list(KERNEL_NAMES.values())

    counts = results.pivot_table(index = rows, columns = "kernel", values = "chosen", aggfunc = ["sum", "count"])
    chosen = counts["sum"].reindex(index = row_order, columns = kernels).astype(int)
    runs = counts["count"].reindex(index = row_order, columns = kernels).astype(int)

    for table in [chosen, runs]:
        table["Total"] = table.sum(axis = 1)
        table.loc["Total"] = table.sum(axis = 0)

    return chosen.astype(str) + " / " + runs.astype(str)

# The tables printed by PrintSelectionCounts for each experiment: (title, rows selected, table rows)
SELECTION_GROUPS = {"real": [("R1", {"experiment": "R1"}, "p"),
                             ("R2", {"experiment": "R2"}, "representation")],
                    "simulated": [("A1", {"experiment": "A1"}, "w"),
                                  ("A2 independent", {"experiment": "A2", "brightness": "independent"}, "representation"),
                                  ("A2 coupled", {"experiment": "A2", "brightness": "coupled"}, "representation")]}

"""
Prints, for each part of an experiment (see SELECTION_GROUPS), each hyperparameter, and each value
of it that was selected at least once in that part, how many runs selected it.

Inputs:
experiment - "real" or "simulated"
path - The path to the experiment's test results
"""
def PrintSelectionCounts(experiment: str,
                         path: Path) -> None:

    results = AddLabels(pd.read_csv(path), latex = False)
    parameters = {"length_factor": ("length factor", lambda a: f"{FormatLength(a)} x l_med"),
                  "lambda": ("lambda", lambda l: f"1e{round(np.log10(l))}")}

    for title, selection, rows in SELECTION_GROUPS[experiment]:
        subset = results[np.logical_and.reduce([results[column] == value for column, value in selection.items()])]

        for parameter, (name, fmt) in parameters.items():
            for value in sorted(subset[parameter].unique(), reverse = True):
                print(f"{title}: {name} = {fmt(value)}")
                print(CountTable(subset, rows, parameter, value).to_string())
                print()
