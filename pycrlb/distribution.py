import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import dataclasses


def get_var(information):
    return np.linalg.inv(information)


@dataclasses.dataclass
class Distribution(ABC):

    @abstractmethod
    def sample(self, n=100):
        pass

    @abstractmethod
    def log_likelihood(self, *args, **kwargs):
        pass

    def get_func(self, *x):
        return lambda *params: torch.mean(self.log_likelihood(*x, *params))

    @property
    def data_size(self):
        return 1

    def __post_init__(self):
        print("HHHHHHHHHHEEERREEEE")
        for field in dataclasses.fields(self):
            if field.type == torch.tensor:
                setattr(self, field.name, torch.as_tensor(getattr(self, field.name)))

    def estimate_fisher_information(self, n=10000000):
        n = n//self.data_size
        x = self.sample(n)
        f = self.get_func(*x)
        params = tuple(getattr(self, field.name) for field in dataclasses.fields(self))
        print(params)
        H = torch.autograd.functional.hessian(f, params)
        H = np.array([[np.array(h) for h in row] for row in H])
        H = H.reshape(H.shape[-1], -1)
        return -np.array(H)  # (self.mu, self.sigma)))/n

    def plot_all_errors(self, color="red", n_params=None, n_iterations=200):
        if n_params is None:
            n_params = sum(len(p) for p in self.params)
        name = self.__class__.__name__
        I = self.estimate_fisher_information()
        I = I[:n_params, :n_params]
        all_var = get_var(I)
        n_samples = [200*i for i in range(1, 10)]
        errors = [self.get_square_errors(n_samples=n, n_iterations=n_iterations, do_plot=False) for n in n_samples]
        self.get_square_errors(n_samples=n_samples[-1], n_iterations=n_iterations, do_plot=True)
        print(errors)
        fig, axes = plt.subplots((n_params+1)//2, 2)
        if (n_params+1)//2 == 1:
            axes = [axes]
        if len(self.params) == 1:
            params = self.params[0]
        else:
            params = self.params
        for i, param in enumerate(params[:n_params]):
            var = all_var[i, i]
            ax = axes[i//2][i % 2]
            ax.axline((0, 0), slope=1/var, color=color, label=name+" CRLB")
            ax.plot(n_samples, 1/np.array(errors)[:, i], color=color, label=name+" errors")
            ax.set_ylabel("1/sigma**2")
            ax.set_xlabel("n_samples")

    def get_square_errors(self, n_samples=1000, n_iterations=1000, do_plot=False):
        estimates = [self.estimate_parameters(n_samples)
                     for _ in range(n_iterations)]
        if len(self.params) == 1:
            true_params = np.array(self.params[0])
            estimates = np.array([np.array(row[0]) for row in estimates])
        else:
            true_params = np.array(self.params)
            estimates = np.array(estimates)
        if do_plot:
            for i, param in enumerate(true_params):
                plt.hist(estimates[:, i])
                plt.axvline(x=param)
                plt.title(f"n={n_samples}")
                plt.show()
        print("E", estimates.mean(axis=0))
        print("T", true_params)
        return ((estimates-true_params)**2).sum(axis=0)/n_iterations
