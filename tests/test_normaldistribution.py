from pycrlb.normaldistribution import NormalDistribution
import pytest


@pytest.fixture
def dist():
    return NormalDistribution(0., 1.)


def test_shape(dist):
    info = dist.estimate_fisher_information(10)
    assert info.shape == (2, 2)
