from .distribution import Distribution
import torch
import numpy as np


class MixtureDistribution(Distribution):
    def __init__(self, distributions, weights):
        self._distributions = distributions
        self.weights = torch.as_tensor(weights)
        n_params = [len(d.params) for d in self._distributions]
        self._param_slices = [slice(end-n, end) for n, end in zip(n_params, np.cumsum(n_params))]

    @property
    def params(self):
        return tuple([p for d in self._distributions for p in d.params] + [self.weights])

    def sample(self, n_samples):
        z = torch.multinomial(self.weights, n_samples, replacement=True)
        counts = torch.bincount(z, minlength=len(self._distributions))
        l = [dist.sample(n)[0] for dist, n in zip(self._distributions, counts)]
        z = torch.concat([torch.full((int(count),), torch.tensor(float(i))) for i, count in enumerate(counts)])
        return [torch.concat(l), z]

    def log_likelihood(self, X, z, *params):
        weights = params[-1]
        log_liks = torch.empty((z.numpy().size,))
        for z_val, dist in enumerate(self._distributions):
            mask = z == z_val
            xs = X[mask]
            log_liks[mask] = torch.log(weights[z_val])+dist.log_likelihood(xs, *params[self._param_slices[z_val]])
        return log_liks

    def estimate_parameters(self):
        pass


class MixtureDistributionX(Distribution):
    def __init__(self, distributions, weights):
        self._distributions = distributions
        self.weights = torch.as_tensor(weights)
        self.params = (torch.hstack([p for d in distributions for p in d.params]+[self._weights]),)#  for d in distributions]+weights)
        self._param_numbers = [len(d.params[0]) for d in distributions]
        self._param_offsets = np.cumsum(self._param_numbers)
        self._n_components = len(self._distributions)

    def log_likelihood(self, x, *params):
        weights = params[0][-self._n_components:]
        ps = [params[0][offset-n:offset] for n, offset
              in zip(self._param_numbers, self._param_offsets)]
        l = [torch.log(w) + d.log_likelihood(x, p)
             for w, d, p in zip(weights, self._distributions, ps)]
        return torch.logsumexp(torch.vstack(l), axis=0)

