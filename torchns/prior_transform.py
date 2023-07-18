import torch
import numpy as np


class UniformPrior:
    def __init__(self, v_ranges: torch.Tensor):
        """
        Unit hypercube transformation for uniform prior
        """
        assert torch.all(v_ranges[:, 1] > v_ranges[:, 0])
        self.dim = len(v_ranges)
        self.v_ranges = v_ranges
        self.v_width = v_ranges[:, 1] - v_ranges[:, 0]
        self.v_bias = v_ranges[:, 0]

    def u_to_v(self, u: torch.Tensor):
        """
        Mapping from uniform distribution on unit hypercube
        to uniform distribution on physical space
        """
        return self.v_bias + self.v_width * u

    def v_to_u(self, v: torch.Tensor):
        """
        Mapping from uniform distribution on physical space
        to uniform distribution on unit hypercube
        """
        return (v - self.v_bias) / self.v_width


class GaussianPrior:
    def __init__(self, mean: torch.Tensor, sigma: torch.Tensor):
        """
        Unit hypercube transformation for Gaussian prior
        """
        self.dim = len(mean)
        self.mean = mean
        self.sigma = sigma

    def u_to_v(self, u: torch.Tensor):
        """
        Mapping from uniform distribution on unit hypercube
        to Gaussian distribution on physical space
        
        NOTE: `torch.nan_to_num` is used to avoid NaNs in the output when proposed
        points from the slice sampler are outside of the hypercube bounds. 
        These points are automatically discarded.
        """
        return torch.nan_to_num(
            torch.erfinv(2 * u - 1) * self.sigma * np.sqrt(2) + self.mean
        )

    def v_to_u(self, v: torch.Tensor):
        """
        Mapping from Gaussian distribution on physical space
        to uniform distribution on unit hypercube
        """
        return torch.erf((v - self.mean) / self.sigma / np.sqrt(2)) / 2 + 0.5
