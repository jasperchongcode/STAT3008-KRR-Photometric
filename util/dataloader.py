import io
import pandas as pd
import requests
from pathlib import Path

N_OBJECTS = 50_000
URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"

QUERY = f"""
SELECT TOP {N_OBJECTS}
  p.objID AS objID, p.dered_u AS u, p.dered_g AS g, p.dered_r AS r, p.dered_i AS i, p.dered_z AS z,
  s.z AS redshift, s.zErr AS redshift_err,
  gi.d4000_n, gi.d4000_n_err
FROM PhotoObj AS p
  JOIN SpecObj AS s      ON s.bestObjID  = p.objID
  JOIN galSpecIndx AS gi ON gi.specObjID = s.specObjID
WHERE s.class = 'GALAXY'
  AND s.sciencePrimary = 1
  AND s.zWarning = 0
  AND gi.d4000_n > 0
  AND s.zErr < 0.01
  AND p.modelMagErr_u BETWEEN 0 AND 0.5
  AND p.modelMagErr_g BETWEEN 0 AND 0.1
  AND p.modelMagErr_r BETWEEN 0 AND 0.1
  AND p.modelMagErr_i BETWEEN 0 AND 0.1
  AND p.modelMagErr_z BETWEEN 0 AND 0.1
  AND (s.specObjID % 15) = 0
"""

"""
Downloads the SDSS galaxy sample with QUERY from the DR18 SkyServer and saves it as a CSV.

Inputs:
path - Where to save the CSV
"""
def DownloadData(path: Path) -> None:

    resp = requests.get(URL, params = {"cmd": QUERY, "format": "csv"}, timeout = 600)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), skiprows = 1)  # skip "#Table1" line
    path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(path, index = False)
    print(f"{len(df)} rows")
