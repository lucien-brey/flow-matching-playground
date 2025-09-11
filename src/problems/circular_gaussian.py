import numpy as np
from scipy.stats import multivariate_normal


class Multivariate2DGaussian:
    def __init__(
        self,
        mu: np.ndarray,
        std: np.ndarray,
    ):
        self.mu = mu
        self.std = std

    def __call__(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ):
        X = np.array([x, y])
        diff = X - self.mu
        exponent = -0.5 * diff.T @ np.linalg.inv(self.std) @ diff
        return np.exp(exponent)


class Gaussian:
    def __init__(
        self,
        mu: float,
        std: float,
    ):
        self.mu = mu
        self.std = std

    def __call__(self, X: np.ndarray):
        return np.exp(-0.5 * (X - self.mu) ** 2 / self.std)


class DOG:
    def __init__(
        self,
        mu_1: np.ndarray,
        mu_2: np.ndarray,
        std_1: np.ndarray,
        std_2: np.ndarray,
    ):
        self.std_1 = std_1
        self.std_2 = std_2
        self.gaussian_1 = multivariate_normal(mean=mu_1, cov=std_1)
        self.gaussian_2 = multivariate_normal(mean=mu_2, cov=std_2)

    def __call__(self, X: np.ndarray):
        return -(
            self.gaussian_1.pdf(X) * 2 * np.pi * np.sqrt(np.linalg.det(self.std_1))
            - self.gaussian_2.pdf(X) * 2 * np.pi * np.sqrt(np.linalg.det(self.std_2))
        )  # need to unnormalize the pdf to get proper shape
