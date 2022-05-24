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
