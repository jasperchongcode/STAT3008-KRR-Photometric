import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

"""
Matern Kernel. Evaluates the kernel between every pair of points in X1 and X2.
Uses the closed forms for half-integer nu, and the Gaussian kernel for nu = np.inf.

Inputs:
X1 - The first set of points, of shape (n, d)
X2 - The second set of points, of shape (m, d)
nu - The smoothness, one of 0.5, 1.5, 2.5 or np.inf
length_scale - The length scale l > 0

Returns the kernel matrix, of shape (n, m)
"""
def Matern(X1: npt.NDArray[np.float64],
           X2: npt.NDArray[np.float64],
           nu: float = 1.5,
           length_scale: float = 1.0) -> npt.NDArray[np.float64]:

    D = cdist(np.atleast_2d(X1), np.atleast_2d(X2)) / length_scale

    if nu == 0.5:
        return np.exp(-D)
    if nu == 1.5:
        return (1 + np.sqrt(3) * D) * np.exp(-np.sqrt(3) * D)
    if nu == 2.5:
        return (1 + np.sqrt(5) * D + 5 * D ** 2 / 3) * np.exp(-np.sqrt(5) * D)
    if nu == np.inf:
        return np.exp(-D ** 2 / 2)

    raise ValueError("nu must be one of 0.5, 1.5, 2.5 or np.inf")