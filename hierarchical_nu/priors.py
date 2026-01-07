from abc import ABCMeta, abstractmethod
from typing import Union, Iterable
import astropy.units as u
from astropy.units.core import UnitConversionError
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


class NormalPrior(PriorDistribution):
    """Normal distribution"""

    def __init__(self, name="normal", mu=0.0, sigma=1.0):
        super().__init__(name=name)

        self._mu = mu
        self._sigma = sigma

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, val: float):
        self._mu = val

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val: float):
        self._sigma = val

    def pdf(self, x):
        return stats.norm(loc=self._mu, scale=self._sigma).pdf(x)

    def sample(self, N):
        return stats.norm(loc=self._mu, scale=self._sigma).rvs(N)

    def to_dict(self, units):
        prior_dict = {}
        prior_dict["name"] = self._name
        prior_dict["mu"] = self._mu
        prior_dict["sigma"] = self._sigma
        prior_dict["units"] = units
        return prior_dict


class LogNormalPrior(PriorDistribution):
    """Log-normal distribution"""

    def __init__(self, name="lognormal", mu=1.0, sigma=1.0):
        super().__init__(name=name)

        self._mu = mu
        self._sigma = sigma

    def pdf(self, x):
        return stats.lognorm(scale=np.exp(self._mu), s=self._sigma).pdf(x)

    def sample(self, N):
        return stats.lognorm(scale=np.exp(self._mu), s=self._sigma).rvs(N)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, val: float):
        self._mu = val

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val: float):
        self._sigma = val

    def to_dict(self, units):
        prior_dict = {}
        prior_dict["name"] = self._name
        prior_dict["mu"] = self._mu
        prior_dict["sigma"] = self._sigma
        prior_dict["units"] = units
        return prior_dict


class UniformPrior(PriorDistribution):
    """
    Uniform prior

    dummy class?
    """

    def __init__(self, name="uniform", xmin=0.0, xmax=np.infty):
        super().__init__(name, xmin=xmin, xmax=xmax)


class LogUniformPrior(PriorDistribution):
    """
    Log-uniform prior, i.e. flat in log(x)
    """

    def __init__(self, name="logflat", xmin=0.0, xmax=np.infty):
        super().__init__(name)

        self._xmin = xmin
        self._xmax = xmax

    @property
    def norm(self):
        return 1.0 / np.log(self.xmax / self.xmin)

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, val: float):
        self._xmin = val

    @property
    def xmax(self):
        return self._xmax

    @xmin.setter
    def xmax(self, val: float):
        self._xmax = val

    def pdf(self, x):
        return 1 / (self.norm * x)

    def sample(self, N):
        stats.loguniform.rvs(self.xmin, self.xmax, size=N)

    def to_dict(self, units):
        prior_dict = {}

        prior_dict["name"] = self._name

        prior_dict["xmin"] = self.xmin

        prior_dict["xmax"] = self.xmax

        prior_dict["units"] = units

        return prior_dict


class ParetoPrior(PriorDistribution):
    """
    Pareto distribution, i.e. x^{-alpha}
    """

    def __init__(self, name="pareto", xmin=1.0, alpha=1.0):
        super().__init__(name=name)

        self._xmin = xmin
        self._alpha = alpha

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, val: float):
        self._xmin = val

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, val: float):
        self._alpha = val

    def pdf(self, x):
        return stats.pareto(b=self._alpha).pdf(x)

    def sample(self, N):
        return stats.pareto(b=self._alpha).rvs(N) * self._xmin

    def to_dict(self, units):
        prior_dict = {}

        prior_dict["name"] = self._name
        prior_dict["xmin"] = self._xmin
        prior_dict["alpha"] = self._alpha
        prior_dict["units"] = units
        return prior_dict


class NoPriorSetError(Exception):
    pass


class Ignorance(PriorDistribution):
    def __init__(self, name="notaprior", **kwargs):
        super().__init__(name=name)

    def pdf(self, x):
        raise NoPriorSetError("Ignorance is bliss, but has no pdf")

    def pdf_logspace(self, x):
        raise NoPriorSetError("Ignorance is your new best friend, but has no pdf")

    def sample(self, N):
        raise NoPriorSetError(
            "I am running out of references, anyway, Ignorance cannot be sampled from"
        )

    def to_dict(self, units):
        return dict(name=self._name, units=units)


class ExponentialGaussianPrior(PriorDistribution):
    def __init__(self, name="exponnorm", mu=0.0, sigma=1.0, lam=1.0):
        super().__init__(name)
        self._mu = mu
        self._sigma = sigma
        self._lam = lam

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, val: float):
        self._mu = val

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, val: float):
        self._sigma = val

    @property
    def lam(self):
        return self._lam

    @lam.setter
    def lam(self, val: float):
        self._lam = val

    def pdf(self, x):
        return stats.exponnorm.pdf(x, 1 / (self.lam * self.sigma), self.mu, self.sigma)

    def sample(self, N):
        return

    def to_dict(self, units):
        priors_dict = {}
        priors_dict["mu"] = self.mu
        priors_dict["sigma"] = self.sigma
        priors_dict["lam"] = self.lam
        priors_dict["units"] = units
        priors_dict["name"] = self.name
        return priors_dict


class PriorDictHandler:
    """
    Class to translate priors from and to dictionaries.
    """

    @classmethod
    def from_dict(cls, prior_dict):
        # Translate "key" into Lumi, Flux or Index
        translate = {
            "L": LuminosityPrior,
            "diffuse_norm": DifferentialFluxPrior,
            "F_atmo": FluxPrior,
            "src_index": IndexPrior,
            "beta_index": IndexPrior,
            "diff_index": IndexPrior,
            "E0_src": EnergyPrior,
            "ang_sys": AngularPrior,
            "eta": EtaPrior,
            "pressure_ratio": PressureRatioPrior,
            "Nex_src": NexPrior,
        }
        prior_name = prior_dict["name"]
        prior = translate[prior_dict["quantity"]]
        units = u.Unit(prior_dict["units"])
        if prior_name == "pareto":
            xmin = prior_dict["xmin"]
            alpha = prior_dict["alpha"]
            return prior(ParetoPrior, xmin=xmin * units, alpha=alpha)
        if prior_name == "logflat":
            xmin = prior_dict["xmin"]
            xmax = prior_dict["xmax"]
            return prior(LogUniformPrior, xmin=xmin * units, xmax=xmax * units)
        elif prior_name == "notaprior":
            return prior(Ignorance)
        mu = np.atleast_1d(prior_dict["mu"])
        sigma = np.atleast_1d(prior_dict["sigma"])
        if prior_name == "normal":
            sigma *= units
            mu *= units
            return prior(NormalPrior, mu=mu[0], sigma=sigma[0])
        elif prior_name == "lognormal":
            return prior(LogNormalPrior, mu=np.exp(mu[0]) * units, sigma=sigma[0])
        elif prior_name == "exponnorm":
            lam = prior_dict["lam"]
            return prior(ExponentialGaussianPrior, mu=mu[0] * units, sigma=sigma[0] * units, lam=lam / units)


class UnitPrior:
    """
    Class to handle priors on parameters carrying a unit.
    """

    def __init__(self, name, **kwargs):
        if name == ParetoPrior:
            xmin = kwargs.get("xmin")
            alpha = kwargs.get("alpha")
            units = kwargs.get("units")
            self._units = units
            self._prior = name(xmin=xmin.to_value(units), alpha=alpha)

        elif name == LogUniformPrior:
            xmin = kwargs.get("xmin")
            xmax = kwargs.get("xmax")
            units = kwargs.get("units")
            self.units = units
            self._prior = name(xmin=xmin.to_value(units), xmax=xmax.to_value(units))
        elif name == ExponentialGaussianPrior:
            mu = kwargs.get("mu")
            sigma = kwargs.get("sigma")
            lam = kwargs.get("lam")
            units = kwargs.get("units")
            self._units = units
            self._prior = name(
                mu=mu.to_value(units),
                sigma=sigma.to_value(units),
                lam=lam.to_value(1 / units),
            )

        else:
            mu = kwargs.get("mu")
            sigma = kwargs.get("sigma")
            units = kwargs.get("units")
            self._units = units

            if name == LogNormalPrior:
                # Check that sigma is a real number
                try:
                    sigma.to_value(1)
                except AttributeError:
                    # Is raised if sigma has no astropy unit attached
                    pass
                # else UnitConversionError is raised
                self._prior = name(mu=np.log(mu.to_value(units)), sigma=sigma)
            elif name == NormalPrior:
                self._prior = name(mu=mu.to_value(units), sigma=sigma.to_value(units))
            elif name == ParetoPrior:
                pass

    def pdf(self, x):
        return self._prior.pdf(x.to_value(self._units))

    def pdf_logspace(self, x):
        return self._prior.pdf_logspace(x.to_value(self._units))

    @property
    def mu(self):
        if isinstance(self._prior, NormalPrior) or isinstance(
            self._prior, ExponentialGaussianPrior
        ):
            return self._prior.mu * self._units
        else:
            return self._prior.mu

    @mu.setter
    def mu(self, val):
        if isinstance(self._prior, NormalPrior) or isinstance(
            self._prior, ExponentialGaussianPrior
        ):
            self._prior.mu = val.to_value(self._units)
        else:
            self._prior.mu = val

    @property
    def sigma(self):
        if isinstance(self._prior, NormalPrior) or isinstance(
            self._prior, ExponentialGaussianPrior
        ):
            return self._prior.sigma * self._units
        else:
            return self._prior.sigma

    @sigma.setter
    def sigma(self, val):
        if isinstance(self._prior, NormalPrior) or isinstance(
            self._prior, ExponentialGaussianPrior
        ):
            self._prior.sigma = val.to_value(self._units)
        else:
            self._prior.sigma = val

    @property
    def xmin(self):
        return self._prior.xmin * self._units

    @xmin.setter
    def xmin(self, val):
        if isinstance(self._prior, ParetoPrior):
            self._prior.xmin = val.to_value(self._units)

    @property
    def alpha(self):
        return self._prior.alpha

    @alpha.setter
    def alpha(self, val):
        self._prior.alpha = val

    @property
    def lam(self):
        return self._prior.lam / self._units

    @lam.setter
    def lam(self, val):
        if isinstance(self._prior, ExponentialGaussianPrior):
            self._prior.lam = val.to_value(1 / self._units)

    @property
    def name(self):
        return self._prior.name

    def to_dict(self, units):
        return self._prior.to_dict(units)


class UnitlessPrior:
    """
    Class to handle priors on unitless parameters.
    """

    UNITS = 1
    UNITS_STRING = "1"

    def __init__(self, name, **kwargs):
        if name == ParetoPrior:
            alpha = kwargs.get("alpha")
            xmin = kwargs.get("xmin")
            self._prior = name(xmin=xmin, alpha=alpha)

        else:
            mu = kwargs.get("mu")
            sigma = kwargs.get("sigma")
            if name == LogNormalPrior:
                self._prior = name(mu=np.log(mu), sigma=sigma)
            elif name == NormalPrior:
                self._prior = name(mu=mu, sigma=sigma)
            elif name == Ignorance:
                self._prior = name()

    @property
    def mu(self):
        return self._prior.mu

    @mu.setter
    def mu(self, val: float):
        self._prior.mu = val

    @property
    def sigma(self):
        return self._prior.sigma

    @sigma.setter
    def sigma(self, val: float):
        self._prior.sigma = val

    @property
    def xmin(self):
        return self._prior.xmin

    @xmin.setter
    def xmin(self, val: float):
        self._prior.xmin = val

    @property
    def alpha(self):
        return self._prior.alpha

    @alpha.setter
    def alpha(self, val: float):
        self._prior.alpha = val

    @property
    def name(self):
        return self._prior.name

    def pdf(self, x):
        return self._prior.pdf(x)

    def pdf_logspace(self, x):
        return self._prior.pdf_logspace(x)

    # Poor man's conditional inheritance
    # copied from https://stackoverflow.com/a/65754897
    # only in case someone tries to do self._mu or similar, then the private attribute should be accessed
    # def __getattr__(self, name):
    #     return self._prior.__getattribute__(name)

    def to_dict(self, units):
        return self._prior.to_dict(units)


class AngularPrior(UnitPrior):
    UNITS = u.deg
    UNITS_STRING = UNITS.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu: Union[u.Quantity[u.deg], None] = 0.2 * u.deg,
        sigma: Union[u.Quantity[u.deg], None] = 0.2 * u.deg,
        lam: Union[u.Quantity[1 / u.deg], None] = None,
    ):
        super().__init__(name, mu=mu, sigma=sigma, lam=lam, units=self.UNITS)

class NexPrior(UnitlessPrior):
    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu=10.,
        sigma=5.,
    ):
        """
        Prior on number of expected events
        """
        super().__init__(
            name,
            mu=mu,
            sigma=sigma,
            units=self.UNITS,
        )

class LuminosityPrior(UnitPrior):
    """
    Source luminosity prior
    """

    UNITS = u.GeV / u.s
    UNITS_STRING = UNITS.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=LogNormalPrior,
        mu: Union[u.Quantity[u.GeV / u.s], None] = 1e49 * u.GeV / u.s,
        sigma: Union[u.Quantity[u.GeV / u.s], u.Quantity[1], None] = 3.0,
        xmin: Union[u.Quantity[u.GeV / u.s], None] = None,
        xmax: Union[u.Quantity[u.GeV / u.s], None] = None,
        alpha: Union[float, None] = None,
    ):
        """
        Converts automatically to log of values, be aware of misuse of notation.
        """
        # This sigma thing is weird due to the log
        super().__init__(
            name,
            mu=mu,
            sigma=sigma,
            xmin=xmin,
            xmax=xmax,
            alpha=alpha,
            units=self.UNITS,
        )


class EnergyPrior(UnitPrior):
    """
    Prior on break energy used in logparabola or pgamma spectrum
    """

    UNITS = u.GeV
    UNITS_STRING = UNITS.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=LogNormalPrior,
        mu: Union[u.Quantity[u.GeV], None] = 1e6 * u.GeV,
        sigma: Union[u.Quantity[u.GeV], u.Quantity[1], None] = 3.0,
    ):
        """
        Converts automatically to log of values, be aware of misuse of notation.
        """
        # This sigma thing is weird due to the log
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class FluxPrior(UnitPrior):
    """
    Prior on number flux dN/dA/dt
    """

    UNITS = 1 / u.m**2 / u.s
    UNITS_STRING = UNITS.unit.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu: u.Quantity[1 / u.m**2 / u.s] = 0.314 / u.m**2 / u.s,
        sigma: Union[u.Quantity[1 / u.m**2 / u.s], u.Quantity[1]] = 0.08 / u.m**2 / u.s,
    ):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class DifferentialFluxPrior(UnitPrior):
    """
    Prior on differential flux dN/dE/dA/dt
    """

    UNITS = 1 / u.GeV / u.m**2 / u.s
    UNITS_STRING = UNITS.unit.to_string()

    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu: u.Quantity[1 / u.GeV / u.m**2 / u.s] = 2.26e-13 / u.GeV / u.m**2 / u.s,
        sigma: Union[u.Quantity[1 / u.GeV / u.m**2 / u.s], u.Quantity[1]] = 0.19e-13
        / u.GeV
        / u.m**2
        / u.s,
    ):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class IndexPrior(UnitlessPrior):
    """
    Spectral index or curvature (for logparabola) prior
    """

    @u.quantity_input
    def __init__(
        self,
        name=NormalPrior,
        mu: float = 2.5,
        sigma: float = 0.5,
    ):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class PressureRatioPrior(UnitlessPrior):
    """
    Prior on the pressure ratio in the corona of Seyfert galaxies
    """

    @u.quantity_input
    def __init__(self, name=Ignorance, mu: float = 0.2, sigma: float = 0.1):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class EtaPrior(UnitlessPrior):
    @u.quantity_input
    def __init__(self, name=Ignorance, mu: float = 40, sigma: float = 10):
        super().__init__(name, mu=mu, sigma=sigma, units=self.UNITS)


class MultiSourcePrior:
    """
    Container base class to handle priors on vector parameters,
    e.g. index prior for multiple point sources. If all sources should
    have the same prior, use the normal classes above.
    """

    def __init__(self, priors):
        """
        :param priors: List of priors
        """

        assert all(isinstance(_._prior, type(priors[0]._prior)) for _ in priors)
        self._priors = priors

    def __getitem__(self, index):
        return self._priors[index]

    def __len__(self):
        return len(self._priors)

    @property
    def name(self):
        return self._priors[0]._prior.name


class MultiSourceLuminosityPrior(MultiSourcePrior, LuminosityPrior):
    def __init__(self, priors: Iterable[LuminosityPrior]):
        assert all(isinstance(_, LuminosityPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        try:
            return np.array([_.mu for _ in self._priors])
        except TypeError:
            return (
                np.array([_.mu.to_value(self.UNITS) for _ in self._priors]) * self.UNITS
            )

    @property
    def sigma(self):
        try:
            return np.array([_.sigma for _ in self._priors])
        except TypeError:
            return (
                np.array([_.sigma.to_value(self.UNITS) for _ in self._priors])
                * self.UNITS
            )

    @property
    def xmin(self):
        try:
            return np.array([_.xmin for _ in self._priors])
        except TypeError:
            return (
                np.array([_.xmin.to_value(self.UNITS) for _ in self._priors])
                * self.UNITS
            )

    @property
    def alpha(self):
        return np.array([_.alpha for _ in self._priors])


class MultiSourceIndexPrior(MultiSourcePrior, IndexPrior):
    def __init__(self, priors: Iterable[IndexPrior]):
        assert all(isinstance(_, IndexPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        return np.array([_.mu for _ in self._priors])

    @property
    def sigma(self):
        return np.array([_.sigma for _ in self._priors])


class MultiSourceEnergyPrior(MultiSourcePrior, EnergyPrior):
    def __init__(self, priors: Iterable[EnergyPrior]):
        assert all(isinstance(_, EnergyPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        return np.array([_.mu for _ in self._priors])

    @property
    def sigma(self):
        return np.array([_.sigma for _ in self._priors])


class MultiSourcePressureRatioPrior(MultiSourcePrior, PressureRatioPrior):
    def __init__(self, priors: Iterable[PressureRatioPrior]):
        assert all(isinstance(_, PressureRatioPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        return np.array([_.mu for _ in self._priors])

    @property
    def sigma(self):
        return np.array([_.sigma for _ in self._priors])


class MultiSourceEtaPrior(MultiSourcePrior, EtaPrior):
    def __init__(self, priors):
        assert all(isinstance(_, EtaPrior) for _ in priors)
        super().__init__(priors)

    @property
    def mu(self):
        return np.array([_.mu for _ in self._priors])

    @property
    def sigma(self):
        return np.array([_.sigma for _ in self._priors])


class Priors(object):
    """
    Container for model priors.

    Defaults to sensible uninformative priors
    on all hyperparameters.
    """

    def __init__(self):
        self._luminosity = LuminosityPrior()

        self.diffuse_flux = DifferentialFluxPrior()

        self.src_index = IndexPrior()

        self.beta_index = IndexPrior(mu=0.0, sigma=0.1)

        self.diff_index = IndexPrior()

        self.atmospheric_flux = FluxPrior()

        self.E0_src = EnergyPrior()

        self.pressure_ratio = PressureRatioPrior()

        self.eta = EtaPrior()

        self.ang_sys = AngularPrior()

        self.Nex_src = NexPrior()

    @property
    def pressure_ratio(self):
        return self._pressure_ratio

    @pressure_ratio.setter
    def pressure_ratio(self, prior: PressureRatioPrior):
        if not isinstance(prior, PressureRatioPrior):
            raise ValueError("Wrong prior type")
        self._pressure_ratio = prior

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, prior: EtaPrior):
        if not isinstance(prior, EtaPrior):
            raise ValueError("Wrong prior type")
        self._eta = prior

    @property
    def ang_sys(self):
        return self._ang_sys

    @ang_sys.setter
    def ang_sys(self, prior: AngularPrior):
        if not isinstance(prior, AngularPrior):
            raise ValueError("Wrong prior type")
        self._ang_sys = prior

    @property
    def luminosity(self):
        return self._luminosity

    @luminosity.setter
    def luminosity(self, prior: LuminosityPrior):
        if not isinstance(prior, LuminosityPrior):
            raise ValueError("Wrong prior type")
        self._luminosity = prior

    @property
    def Nex_src(self):
        return self._nex_prior

    @Nex_src.setter
    def Nex_src(self, prior: NexPrior):
        if not isinstance(prior, NexPrior):
            raise ValueError("Wrong prior type")
        self._nex_prior = prior

    @property
    def diffuse_flux(self):
        return self._diffuse_flux

    @diffuse_flux.setter
    def diffuse_flux(self, prior: DifferentialFluxPrior):
        if not isinstance(prior, DifferentialFluxPrior):
            raise ValueError("Wrong prior type")
        self._diffuse_flux = prior

    @property
    def src_index(self):
        return self._src_index

    @src_index.setter
    def src_index(self, prior: IndexPrior):
        if not isinstance(prior, IndexPrior):
            raise ValueError("Wrong prior type")
        self._src_index = prior

    @property
    def beta_index(self):
        return self._beta_index

    @beta_index.setter
    def beta_index(self, prior: IndexPrior):
        if not isinstance(prior, IndexPrior):
            raise ValueError("Wrong prior type")
        self._beta_index = prior

    @property
    def E0_src(self):
        return self._E0_src

    @E0_src.setter
    def E0_src(self, prior: EnergyPrior):
        if not isinstance(prior, EnergyPrior):
            raise ValueError("Wrong prior type")
        self._E0_src = prior

    @property
    def diff_index(self):
        return self._diff_index

    @diff_index.setter
    def diff_index(self, prior: IndexPrior):
        if not isinstance(prior, IndexPrior):
            raise ValueError("Wrong prior type")
        self._diff_index = prior

    @property
    def atmospheric_flux(self):
        return self._atmospheric_flux

    @atmospheric_flux.setter
    def atmospheric_flux(self, prior: FluxPrior):
        if not isinstance(prior, FluxPrior):
            raise ValueError("Wrong prior type")
        self._atmospheric_flux = prior

    def to_dict(self):
        priors_dict = {}

        priors_dict["L"] = self.luminosity

        priors_dict["diffuse_norm"] = self.diffuse_flux

        priors_dict["F_atmo"] = self.atmospheric_flux

        priors_dict["src_index"] = self.src_index

        priors_dict["beta_index"] = self.beta_index

        priors_dict["diff_index"] = self.diff_index

        priors_dict["beta_index"] = self.beta_index

        priors_dict["E0_src"] = self.E0_src

        priors_dict["eta"] = self.eta

        priors_dict["pressure_ratio"] = self.pressure_ratio

        priors_dict["ang_sys"] = self.ang_sys

        priors_dict["Nex_src"] = self.Nex_src

        return priors_dict

    def save(self, file_name: str):
        """
        Save priors to file
        :param file_name: filename
        """

        with h5py.File(file_name, "w") as f:
            self._writeto(f)

    def _writeto(self, f):
        priors_dict = self.to_dict()

        def create_dataset(g, prior):
            for key, value in prior.to_dict(prior.UNITS_STRING).items():
                g.create_dataset(key, data=value)

        for key, value in priors_dict.items():
            g = f.create_group(key)

            if isinstance(value, MultiSourcePrior):
                for c, prior in enumerate(value):
                    sub_g = g.create_group(f"ps_{c}")
                    create_dataset(sub_g, prior)

            else:
                create_dataset(g, value)

    def addto(self, file_name: str, group_name: str):
        """
        Save priors to existing file
        :param file_name: existing h5 file
        :param group_name: group name of prior data
        """
        with h5py.File(file_name, "r+") as f:
            g = f.create_group(group_name)

            self._writeto(g)

    @classmethod
    def from_file(cls, file_name: str):
        """
        Load prior data from h5 file
        :param file_name: filename
        """

        with h5py.File(file_name, "r") as f:
            return cls._load_from(f)

    @classmethod
    def from_group(cls, file_name: str, group_name: str):
        """
        Load prior data from h5 file combined with other data
        :param file_name: filename
        :param group_name: group name of prior data
        """

        with h5py.File(file_name, "r") as f:
            g = f[group_name]

            return cls._load_from(g)

    @classmethod
    def _load_from(cls, f):
        priors_dict = {}

        def make_dict_entry(d, arg):
            for k, v in arg.items():
                if k == "name":
                    # name should be replaced by LuminosityPrior etc.
                    d[k] = v[()].decode("ascii")

                else:
                    d[k] = v[()]
            return d

        for key, value in f.items():
            prior_dict = {}
            prior_dict["quantity"] = key

            # for k, v in value.items():
            try:
                prior_dict = make_dict_entry(prior_dict, value)
                priors_dict[key] = PriorDictHandler.from_dict(prior_dict)
            except TypeError:
                # Found the multi source prior

                container = []
                for _, b in value.items():
                    prior_dict = {"quantity": key}
                    prior_dict = make_dict_entry(prior_dict, b)
                    container.append(PriorDictHandler.from_dict(prior_dict))
                if key == "src_index":
                    priors_dict[key] = MultiSourceIndexPrior(container)
                elif key == "beta_index":
                    priors_dict[key] = MultiSourceIndexPrior(container)
                elif key == "E0_src":
                    priors_dict[key] = MultiSourceEnergyPrior(container)
                elif key == "L":
                    priors_dict[key] = MultiSourceLuminosityPrior(container)
                elif key == "ang_sys":
                    # should not happen
                    raise ValueError("There is only one systematic angular uncertainty")
        return cls.from_dict(priors_dict)

    @classmethod
    def from_dict(cls, priors_dict):
        priors = cls()

        priors.luminosity = priors_dict["L"]

        priors.diffuse_flux = priors_dict["diffuse_norm"]

        priors.atmospheric_flux = priors_dict["F_atmo"]

        priors.src_index = priors_dict["src_index"]

        priors.diff_index = priors_dict["diff_index"]
        
        try:
            # Backwards compatiblity
            priors.beta_index = priors_dict["beta_index"]
        except KeyError:
            pass

        try:
            priors.eta = priors_dict["eta"]
        except KeyError:
            pass

        try:
            priors.pressure_ratio = priors_dict["pressure_ratio"]
        except KeyError:
            pass

        try:
            priors.ang_sys = priors_dict["ang_sys"]
        except KeyError:
            pass

        try:
            priors.E0_src = priors_dict["E0_src"]
        except KeyError:
            pass

        try:
            priors.Nex_src = priors_dict["Nex_src"]
        except KeyError:
            pass

        return priors
