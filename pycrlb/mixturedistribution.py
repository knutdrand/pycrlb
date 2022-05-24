from .distribution import Distribution
import torch
import numpy as np


class MixtureDistribution(Distribution):
    def __init__(self, distributions, weights):
        self._distributions = distributions
        self._weights = torch.as_tensor(weights)
        self.params = (torch.hstack([p for d in distributions for p in d.params]+[self._weights]),)#  for d in distributions]+weights)
        self._param_numbers = [len(d.params[0]) for d in distributions]
        self._param_offsets = np.cumsum(self._param_numbers)
        self._n_components = len(self._distributions)

    @property
    def param_dict(self):
        return {key: val for key, val in vars(self) if not key.startswith("-")}

    def log_likelihood(self, x, *params):
        weights = params[0][-self._n_components:]
        ps = [params[0][offset-n:offset] for n, offset
              in zip(self._param_numbers, self._param_offsets)]
        l = [torch.log(w) + d.log_likelihood(x, p)
             for w, d, p in zip(weights, self._distributions, ps)]
        return torch.logsumexp(torch.vstack(l), axis=0)

    def sample(self, n_samples):
        z = torch.multinomial(self._weights, n_samples, replacement=True)
        counts = torch.bincount(z, minlength=len(self._distributions))
        l = [dist.sample(n)[0] for dist, n in zip(self._distributions, counts)]
        return [torch.vstack(l)]
