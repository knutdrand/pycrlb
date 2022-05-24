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

    def estimate_parameters(self, n):
        X, = self.sample(n)
        mu = torch.mean(X, axis=0)
        d = X-mu
        s = (d**2).mean()
        return mu.numpy(), np.sqrt(s.numpy())


@dataclass
class MultiNormalDistribution(Distribution):
    mu: torch.tensor = torch.tensor([0., 1.])
    sigma: torch.tensor = torch.tensor([[1., 0.], [0., 2.]])

    def sample(self, n=1):
        data = np.random.multivariate_normal(self.mu, self.sigma, n)
        return [torch.as_tensor(data)]
        return [torch.normal(self.mu, self.sigma, (n,))]

    def log_likelihood(self, x, mu, sigma):
        d = (x-mu)[..., None]
        a = -1/2*torch.log(torch.linalg.det(2*np.pi*sigma))
        b = d.swapaxes(-1, -2)@torch.linalg.inv(sigma)@d/2
        return a-b

    def estimate_parameters(self, n):
        X, = self.sample(n)
        mu = torch.mean(X, axis=0)
        d = X-mu
        cov = torch.mean(d[..., None, :]*d[..., None], axis=0)
        return mu.numpy(), cov.numpy()
