import numpy as np
from abc import ABCMeta


class PriorDistribution(metaclass=ABCMeta):
    """
    Abstract base class for prior
    distributions.
    """

    def __init__(self, name="prior"):

        self._name = name

    @property
    def name(self):

        return self._name


class NormalPrior(PriorDistribution):
    def __init__(self, name="normal", mu=0.0, sigma=1.0):

        super().__init__(name=name)

        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self):

        return self._mu

    @property
    def sigma(self):

        return self._sigma


class LogNormalPrior(NormalPrior):
    def __init__(self, name="lognormal", mu=1.0, sigma=1.0):

        super().__init__(name=name, mu=mu, sigma=sigma)


class ParetoPrior(PriorDistribution):
    def __init__(self, name="pareto", xmin=1.0, alpha=1.0):

        super().__init__(self, name=name)

        self._xmin = xmin
        self._alpha = alpha

    @property
    def xmin(self):

        return self._xmin

    @property
    def alpha(self):

        return self._alpha


class Priors(object):
    """
    Container for model priors.

    Defaults to sensible uninformative priors
    on all hyperparameters.
    """

    def __init__(self):

        self._luminosity = LogNormalPrior(mu=np.log(1e45), sigma=10.0)

        self._diffuse_flux = LogNormalPrior(mu=np.log(1e-8), sigma=5.0)

        self._src_index = NormalPrior(mu=2.0, sigma=1.5)

        self._diff_index = NormalPrior(mu=2.0, sigma=1.5)

        self._atmospheric_flux = LogNormalPrior(mu=np.log(1e-8), sigma=3.0)

    @property
    def luminosity(self):

        return self._luminosity

    @luminosity.setter
    def luminosity(self, prior: PriorDistribution):

        self._luminosity = prior

    @property
    def diffuse_flux(self):

        return self._diffuse_flux

    @diffuse_flux.setter
    def diffuse_flux(self, prior: PriorDistribution):

        self._diffuse_flux = prior

    @property
    def src_index(self):

        return self._src_index

    @src_index.setter
    def src_index(self, prior: PriorDistribution):

        self._src_index = prior

    @property
    def diff_index(self):

        return self._diff_index

    @diff_index.setter
    def diff_index(self, prior: PriorDistribution):

        self._diff_index = prior

    @property
    def atmospheric_flux(self):

        return self._atmospheric_flux

    @atmospheric_flux.setter
    def atmospheric_flux(self, prior: PriorDistribution):

        self._atmospheric_flux = prior


class InformativePriors(Priors):
    """
    Standard informative priors
    """

    def __init__(self):

        super().__init__()

        self._luminosity = ParetoPrior(xmin=1e46, alpha=1.0)

        self._src_index = NormalPrior(mu=2.0, sigma=0.5)
