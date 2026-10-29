import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from pathlib import Path
from scipy.special import ndtr
from astropy.io import fits
from util.evaluation import BANDS

Spectrum = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

"""
Generates a toy rest-frame spectrum (flux per unit frequency): a flat continuum
with a smoothed step at lam_0,
s(lam) = 1 + (B - 1) * Phi((lam - lam_0) / w)

Inputs:
lam_0 - The break wavelength in Angstrom
B - The break strength. s is approximately 1 blueward of the break and B redward
w - The break width in Angstrom, w > 0

Returns the spectrum s as a function of wavelength
"""
def GenerateSpectrum(lam_0: float = 4000.0,
                     B: float = 1.6,
                     w: float = 25.0) -> Spectrum:

    def spectrum(lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 + (B - 1) * ndtr((lam - lam_0) / w)

    return spectrum

"""
Redshifts a spectrum. Light observed at lam was emitted at lam / (1 + z), so the
observed spectrum is s(lam / (1 + z)). Any overall amplitude is left to AddBrightness.

Inputs:
spectrum - The rest-frame spectrum
z - The redshift

Returns the observed spectrum as a function of observed wavelength
"""
def Redshift(spectrum: Spectrum,
             z: float) -> Spectrum:

    def redshifted(lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return spectrum(lam / (1 + z))

    return redshifted

"""
Loads the SDSS filter response curves from the FITS file. Uses the 'respt' column,
the response for a point source at 1.3 airmass.

Inputs:
path - The path to sdss_filter_curves.fits

Returns a dictionary mapping each band to (wavelength, response), wavelength in Angstrom
"""
def LoadFilterCurves(path: Path) -> dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:

    filters = {}
    with fits.open(path) as hdul:
        for k, band in enumerate(BANDS, start = 1):
            table = hdul[k].data
            wavelength = np.asarray(table["wavelength"], dtype = np.float64)
            response = np.asarray(table["respt"], dtype = np.float64)
            filters[band] = (wavelength, response)

    return filters

"""
Calculates the synthetic magnitude of a spectrum through a single response curve,
m = -2.5 log10( int s(lam) T(lam) dlam/lam / int T(lam) dlam/lam )
The response is linearly interpolated onto a grid with the given spacing, and both
integrals are evaluated with the trapezoidal rule.

Inputs:
spectrum - The observed spectrum
wavelength - The wavelengths at which the response is tabulated, in Angstrom
response - The response curve T at each wavelength
spacing - The grid spacing in Angstrom

Returns the magnitude through this response curve
"""
def SyntheticPhotometry(spectrum: Spectrum,
                        wavelength: npt.NDArray[np.float64],
                        response: npt.NDArray[np.float64],
                        spacing: float = 2.0) -> float:

    grid = np.arange(wavelength[0], wavelength[-1] + spacing, spacing)
    T = np.interp(grid, wavelength, response, left = 0, right = 0)

    numerator = np.trapezoid(spectrum(grid) * T / grid, grid)
    denominator = np.trapezoid(T / grid, grid)

    return float(-2.5 * np.log10(numerator / denominator))

"""
Calculates the synthetic magnitudes of a spectrum through every response curve.

Inputs:
spectrum - The observed spectrum
filters - A dictionary mapping each band to (wavelength, response), as from LoadFilterCurves
spacing - The grid spacing in Angstrom

Returns the magnitudes in the order of filters, of shape (number of bands)
"""
def Photometry(spectrum: Spectrum,
               filters: dict[str, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]],
               spacing: float = 2.0) -> npt.NDArray[np.float64]:

    magnitudes = []
    for wavelength, response in filters.values():
        magnitudes.append(SyntheticPhotometry(spectrum, wavelength, response, spacing))

    return np.array(magnitudes)

"""
Sets the brightness of each galaxy, its component along the direction (1, ..., 1).
The shape magnitudes are centred so their mean over bands is 0, then a single offset Delta is
added to every band, m_b = m_b^shape - mean(m^shape) + Delta. The mean magnitude of each galaxy
is then exactly Delta, and colours are unaffected.

Modes:
independent - Delta ~ U(15.44, 19.46), the 2.5-97.5% range of the mean magnitude in the SDSS sample
coupled - Delta = a + b log10(z) + eps with eps ~ N(0, sigma^2), fitted by regressing the mean
          magnitude on log10(z) in the SDSS sample

Inputs:
magnitudes - The shape magnitudes, of shape (n, number of bands)
z - The redshift of each galaxy, of shape (n)
mode - Either "independent" or "coupled"
rng - The random number generator

Returns the magnitudes with brightness set, of shape (n, number of bands)
"""
def AddBrightness(magnitudes: npt.NDArray[np.float64],
                  z: npt.NDArray[np.float64],
                  mode: str = "independent",
                  rng: np.random.Generator | None = None) -> npt.NDArray[np.float64]:

    if rng is None:
        rng = np.random.default_rng()

    magnitudes = np.atleast_2d(magnitudes)
    z = np.atleast_1d(z)
    assert (len(magnitudes) == len(z))
    n = len(z)

    if mode == "independent":
        delta = rng.uniform(15.44, 19.46, size = n)
    elif mode == "coupled":
        a, b, sigma = 19.76, 2.20, 0.69
        delta = a + b * np.log10(z) + rng.normal(0, sigma, size = n)
    else:
        raise ValueError("mode must be 'independent' or 'coupled'")

    centred = magnitudes - np.mean(magnitudes, axis = 1, keepdims = True)
    return centred + delta[:, None]
