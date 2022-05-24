import torch
import numpy as np
from .distribution import Distribution
from dataclasses import dataclass


@dataclass
class NormalDistribution(Distribution):
    mu: torch.tensor = torch.tensor(0.)
    sigma: torch.tensor = torch.tensor(1.)

    def sample(self, n=1):
        return [torch.normal(self.mu, self.sigma, (n,))]

    def log_likelihood(self, x, mu, sigma):
        return torch.log(
            1/torch.sqrt(2*np.pi*sigma**2)) - (x-mu)**2/(2*sigma**2)


@dataclass
class MultiNormalDistribution(Distribution):
    mu: torch.tensor = torch.tensor(0.)
    sigma: torch.tensor = torch.tensor(1.)

    def sample(self, n=1):
        data = np.random.multivariate_normal(self.mu, self.sigma, n)
        return [torch.as_tensor(data)]
        return [torch.normal(self.mu, self.sigma, (n,))]

    def log_likelihood(self, x, mu, sigma):
        d = (x-mu)[..., None]
        print(x.shape, mu.shape, d.shape, sigma.shape)
        a = -1/2*torch.log(torch.linalg.det(2*np.pi*sigma))
        b = d.swapaxes(-1, -2)@torch.linalg.inv(sigma)@d/2
        return a-b
