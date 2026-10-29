import numpy as np
import numpy.typing as npt
from collections.abc import Callable, Iterable
import scipy as sp
from typing import Any
from models.matern import Matern

Kernel = Callable[[Any, Any], npt.NDArray[np.float64]]

"""
Calculates the norm of a function f in H_k.

Inputs:
kernel - The kernel function
X - The set of points X used to define H_0. i.e. f is in span{K_x(y) : x in X}
C - The set of coefficients defining f. i.e. f = sum(c_i * K_xi(y))
"""
def Norm(kernel: Kernel,
         X: Iterable[Any],
         C: Iterable[np.float64]) -> np.float64:

    assert (len(X) == len(C))

    C = np.asarray(C)
    return C @ kernel(X, X) @ C

"""
Calculates f(x) for some f in H_0 at each point x in a set of points.

Inputs:
kernel - The kernel function
x - The set of points at which to evaluate f, of shape (m, d)
X - The set of points X used to define H_0. i.e. f is in span{K_x(y) : x in X}
C - The set of coefficients defining f. i.e. f = sum(c_i * K_xi(y))
"""
def Calculate(kernel: Kernel,
              x: Any, 
              X: Iterable[Any],
              C: Iterable[np.float64]) -> npt.NDArray[np.float64]:

    assert (len(X) == len(C))

    return kernel(x, X) @ C

"""
Kernel Ridge Regression Class.
"""
class KernelRidgeRegression():

    def __init__(self) -> None:
        self.alpha = None
        self.kernel = None
        self.X = None
        self.y_mean = None

    """
    Fits the model to the dataset. Finds a function f to minimise:
    mean((y_i - f(x_i))**2) + l * Norm(f)
    y is centred before fitting, and the mean is added back to predictions.
    
    Inputs:
    X - The training data X, of shape (n, d)
    y - The training target y, of shape (n)
    l - The ridge regression penalty
    kernel - The kernel function
    K - A pre-computed kernel matrix, of shape (n, n) - to remove duplicate calculations
    """
    def fit(self,
            X: Iterable[Any],
            y: npt.NDArray[np.float64],
            l: np.float64 = 1e-6,
            kernel: Kernel = Matern,
            K: npt.NDArray[np.float64] | None = None) -> np.float64:

        self.kernel = kernel
        self.X = np.asarray(X)

        assert (len(X) == y.size)
        n = len(X)

        self.y_mean = np.mean(y)

        if K is None:
            K = kernel(self.X, self.X)
        self.alpha = sp.linalg.solve(K + l * n * np.identity(n), y - self.y_mean, assume_a = 'pos')

        return np.mean((y - self.y_mean - K @ self.alpha) ** 2)
    
    """
    Predicts y for a given input X. Predictions are computed in batches so the
    full (n, m) cross-kernel matrix is never held in memory at once.
    
    Inputs:
    X - The set of data to predict from, of shape (n, d)
    batch_size - The number of points to predict at once
    """
    def predict(self,
                X: Iterable[Any],
                batch_size: int = 1000) -> npt.NDArray[np.float64]:

        X = np.asarray(X)
        y_pred = np.empty(len(X))

        for start in range(0, len(X), batch_size):
            end = start + batch_size
            y_pred[start:end] = Calculate(self.kernel, X[start:end], self.X, self.alpha)

        return y_pred + self.y_mean
    
    """
    Returns the model MSE given an input X, and a true value y. 
    
    Inputs:
    X - The set of data to predict from, of shape (n, d)
    y - The true output values, of shape (n)
    """
    def score(self, 
              X: Iterable[Any], 
              y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:

        y_pred = self.predict(X)
        assert (y.shape == y_pred.shape)

        return np.mean((y - y_pred) ** 2)
        
        
