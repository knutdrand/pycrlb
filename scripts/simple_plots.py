import matplotlib.pyplot as plt
from importlib import reload
import pycrlb.normaldistribution as nd
import pycrlb.distribution as d
reload(d)
reload(nd)
dist = nd.MultiNormalDistribution([0., 1.], np.diag([1., 2.]))
dist.plot_all_errors()
plt.show()
