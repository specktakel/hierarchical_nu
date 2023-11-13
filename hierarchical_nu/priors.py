from abc import ABCMeta, abstractmethod
from typing import Union
import astropy.units as u
import numpy as np
from scipy import stats
import h5py


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

    @abstractmethod
    def pdf(self, x):
        pass

    def pdf_logspace(self, x):
        return self.pdf(x) * x * np.log(10)

    @abstractmethod
    def sample(self, N):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    def from_dict(cls, prior_dict):
        name = prior_dict["name"]

        if name == "normal":
            prior = NormalPrior(**prior_dict)

        elif name == "lognormal":
            prior = LogNormalPrior(**prior_dict)

        elif name == "pareto":
            prior = ParetoPrior(**prior_dict)

        else:
            raise ValueError("Prior distribution type not recognised")

        return prior


class NormalPrior(PriorDistribution):
    def __init__(self, name="normal", mu=0.0, sigma=1.0):
        if isinstance(mu, u.Quantity) or isinstance(sigma, u.Quantity):
            assert mu.is_equivalent(sigma)

        super().__init__(name=name)

        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    def pdf(self, x):
        return stats.norm(loc=self._mu, scale=self._sigma).pdf(x)

    def sample(self, N):
        return stats.norm(loc=self._mu, scale=self._sigma).rvs(N)

    def to_dict(self):
        prior_dict = {}

        prior_dict["name"] = self._name

        prior_dict["mu"] = self._mu

        prior_dict["sigma"] = self._sigma

        return prior_dict


class LogNormalPrior(NormalPrior):
    def __init__(self, name="lognormal", mu=1.0, sigma=1.0):
        super().__init__(name=name, mu=mu, sigma=sigma)

    def pdf(self, x):
        return stats.lognorm(scale=np.exp(self._mu), s=self._sigma).pdf(x)

    def sample(self, N):
        return stats.lognorm(scale=np.exp(self._mu), s=self._sigma).rvs(N)


class ParetoPrior(PriorDistribution):
    def __init__(self, name="pareto", xmin=1.0, alpha=1.0):
        super().__init__(name=name)

        self._xmin = xmin
        self._alpha = alpha

    @property
    def xmin(self):
        return self._xmin

    @property
    def alpha(self):
        return self._alpha

    def pdf(self, x):
        return stats.pareto(b=self._alpha).pdf(x)

    def sample(self, N):
        return stats.pareto(b=self._alpha).rvs(N) * self._xmin

    def to_dict(self):
        prior_dict = {}

        prior_dict["name"] = self._name

        prior_dict["xmin"] = self._xmin

        prior_dict["alpha"] = self._alpha

        return prior_dict


class UnitPrior:
    def __init__(self, name, **kwargs):
        if name == ParetoPrior:
            raise NotImplementedError

        else:
            mu = kwargs.get("mu")
            sigma = kwargs.get("sigma")
            units = kwargs.get("units")
            if name == LogNormalPrior:
                # In this case sigma is a real number
                self._prior = name(mu=np.log(mu.to_value(units)), sigma=sigma)
            elif name == NormalPrior:
                self._prior = name(mu=mu.to_value(units), sigma=sigma.to_value(units))

    # Poor man's conditional inheritance
    # copied from https://stackoverflow.com/a/65754897
    def __getattr__(self, name):
        return self._prior.__getattribute__(name)


class LuminosityPrior(UnitPrior):
    UNITS = u.GeV / u.s

    # @u.quantity_input
    def __init__(
        self,
        name=LogNormalPrior,
        mu: u.GeV / u.s = 1e49 * u.GeV / u.s,
        sigma: Union[u.Quantity[u.GeV / u.s], float] = 3.0,
    ):
        """
        Converts automatically to log of values, be aware of misuse of notation.
        """
        # This sigma thing is weird due to the log
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class FluxPrior(UnitPrior):
    UNITS = 1 / u.m**2 / u.s

    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu: 1 / u.m**2 / u.s = 0.314 / u.m**2 / u.s,
        sigma: 1 / u.m**2 / u.s = 0.08 / u.m**2 / u.s,
    ):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class Priors(object):
    """
    Container for model priors.

    Defaults to sensible uninformative priors
    on all hyperparameters.
    """

    def __init__(self):
        # self.luminosity = LogNormalPrior(mu=np.log(1e50), sigma=10.0)
        self._luminosity = LuminosityPrior()

        self.diffuse_flux = LogNormalPrior(mu=np.log(1e-7), sigma=1.0)

        self.src_index = IndexPrior()

        self.diff_index = IndexPrior()

        self.atmospheric_flux = FluxPrior(mu=np.log(1e-5), sigma=1.0)

    @property
    def luminosity(self):
        return self._luminosity

    @luminosity.setter
    def luminosity(self, prior: PriorDistribution):
        units = u.GeV / u.s
        if not isinstance(prior, ParetoPrior):
            prior._mu = prior._mu.to(units)
            sigma = prior.sigma

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

    def to_dict(self):
        priors_dict = {}

        priors_dict["L"] = self._luminosity

        priors_dict["F_diff"] = self._diffuse_flux

        priors_dict["F_atmo"] = self._atmospheric_flux

        priors_dict["src_index"] = self._src_index

        priors_dict["diff_index"] = self._diff_index

        return priors_dict

    def save(self, file_name: str):
        with h5py.File(file_name, "w") as f:
            self._writeto(f)

    def _writeto(self, f):
        priors_dict = self.to_dict()

        for key, value in priors_dict.items():
            g = f.create_group(key)

            for key, value in value.to_dict().items():
                g.create_dataset(key, data=value)

    def addto(self, file_name: str, group_name: str):
        with h5py.File(file_name, "r+") as f:
            g = f.create_group(group_name)

            self._writeto(g)

    @classmethod
    def from_file(cls, file_name: str):
        with h5py.File(file_name, "r") as f:
            return cls._load_from(f)

    @classmethod
    def from_group(cls, file_name: str, group_name: str):
        with h5py.File(file_name, "r") as f:
            g = f[group_name]

            return cls._load_from(g)

    @classmethod
    def _load_from(cls, f):
        priors_dict = {}

        for key, value in f.items():
            prior_dict = {}

            for k, v in value.items():
                if k == "name":
                    prior_dict[k] = v[()].decode("ascii")

                else:
                    prior_dict[k] = v[()]

            priors_dict[key] = PriorDistribution.from_dict(prior_dict)

        return cls.from_dict(priors_dict)

    @classmethod
    def from_dict(cls, priors_dict):
        priors = cls()

        priors.luminosity = priors_dict["L"]

        priors.diffuse_flux = priors_dict["F_diff"]

        priors.atmospheric_flux = priors_dict["F_atmo"]

        priors.src_index = priors_dict["src_index"]

        priors.diff_index = priors_dict["diff_index"]

        return priors
