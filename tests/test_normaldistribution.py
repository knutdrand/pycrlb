from pycrlb.normaldistribution import (
    NormalDistribution,
    MultiNormalDistribution,
    )
import numpy as np
import pytest


@pytest.fixture
def dist():
    return NormalDistribution(0., 1.)


@pytest.fixture
def multi_dist():
    return MultiNormalDistribution([0., 1.], np.diag([1., 2.]))


def test_shape(dist):
    info = dist.estimate_fisher_information(10)
    assert info.shape == (2, 2)


def test_multishape(multi_dist):
    info = multi_dist.estimate_fisher_information(10)
    assert info.shape == (6, 6)


def test_multi_estimate(multi_dist):
    mu, cov = multi_dist.estimate_parameters(10)
    assert mu.shape == (2,)
    assert cov.shape == (2, 2)


def test_multi_sq_error(multi_dist):
    mu_err, sigma_err = multi_dist.get_square_errors(10, 10)
    assert mu_err.shape == (2,)
    assert sigma_err.shape == (2, 2)


def test_plot_all_erros_shape(multi_dist):
    multi_dist.plot_all_errors()
