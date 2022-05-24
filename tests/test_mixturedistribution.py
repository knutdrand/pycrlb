from pycrlb.normaldistribution import (
    NormalDistribution,
    MultiNormalDistribution,
    )
from pycrlb.mixturedistribution import MixtureDistribution
import numpy as np
import pytest


@pytest.fixture
def dist():
    return MixtureDistribution([
        NormalDistribution(0., 1.),
        NormalDistribution(1., 1.)],
                               [0.4, 0.6])


def test_shape(dist):
    X, z = dist.sample(10)
    assert X.shape == (10,)
    assert z.shape == (10,)
    log_liks = dist.log_likelihood(X, z, *dist.params)
    assert log_liks.numpy().shape == (10,)
    info = dist.estimate_fisher_information(10)
    assert info.shape == (6, 6)
    mu_1, s_1, mu_2, s_2, w = dist.estimate_parameters(10)
    assert np.atleast_1d(mu_1).shape == (1, )
    assert np.atleast_1d(s_1).shape == (1, )
    mu_err, sigma_err, mu_2_err, s_2_err, w_err = dist.get_square_errors(10, 10)
    assert np.atleast_1d(mu_err).shape == (1, )
    assert np.atleast_1d(sigma_err).shape == (1, )
