from ..priors import (
    Priors,
    LogNormalPrior,
    NormalPrior,
    ParetoPrior,
    LuminosityPrior,
    IndexPrior,
    FluxPrior,
    DifferentialFluxPrior,
    NexPrior,
    EnergyPrior,
    MultiSourceEnergyPrior,
    MultiSourceIndexPrior,
    MultiSourceLuminosityPrior,
    EtaPrior,
    MultiSourceEtaPrior,
    PressureRatioPrior,
    MultiSourcePressureRatioPrior,
    Ignorance,
)
from ..utils.config import HierarchicalNuConfig
from ..source.source import (
    Sources,
    PointSource,
    SourceFrame,
    DetectorFrame,
)
from ..source.parameter import Parameter, ParScale
from ..detector.icecube import Refrigerator
from ..utils.roi import (
    ROIList,
    CircularROI,
    NorthernSkyROI,
    FullSkyROI,
    RectangularROI,
)
from ..detector.input import mceq
from ..utils.lifetime import LifeTime
from ..events import Events

import omegaconf

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import omegaconf


class ConfigParser:

    @classmethod
    def check_units(cls, entry, unit):
        if isinstance(entry, omegaconf.listconfig.ListConfig):
            for _ in entry:
                cls.check_units(_, unit)  # I can do recursion lol
        else:
            try:
                u.Quantity(entry).to(unit)
            except u.UnitConversionError as e:
                raise e
        return

    def __init__(self, hnu_config: HierarchicalNuConfig):
        self._hnu_config = hnu_config

    @property
    def sources(self):

        parameter_config = self._hnu_config["parameter_config"]
        share_L = parameter_config["share_L"]
        share_src_index = parameter_config["share_src_index"]

        for k, v in dict(parameter_config).items():
            if "E" in k and k != "Emin_det_eq":
                self.check_units(v, u.GeV)

            if "L" == k:
                self.check_units(v, u.GeV / u.s)

            if k == "src_dec" or k == "src_ra":
                self.check_units(v, u.deg)

            if "F_atmo" in k:
                self.check_units(v, 1 / u.m**2 / u.s)

            if "diff_norm" in k:
                self.check_units(v, 1 / u.GeV / u.m**2 / u.s)

        Parameter.clear_registry()
        index = []
        beta = []
        E0_src = []
        eta = []
        P = []

        if (
            parameter_config["source_type"] == "power-law"
            or parameter_config["source_type"] == "twice-broken-power-law"
            or parameter_config["source_type"] == "logparabola"
        ):
            if "src_index" in parameter_config["fit_params"] and share_src_index:
                index.append(
                    Parameter(
                        parameter_config["src_index"][0],
                        "src_index",
                        False,
                        parameter_config["src_index_range"],
                    )
                )
            else:
                for c, idx in enumerate(parameter_config["src_index"]):
                    name = f"ps_{c}_src_index"
                    index.append(
                        Parameter(
                            idx,
                            name,
                            not "src_index" in parameter_config["fit_params"],
                            parameter_config["src_index_range"],
                        )
                    )

        if parameter_config["source_type"] == "logparabola":
            if "beta_index" in parameter_config["fit_params"] and share_src_index:
                beta.append(
                    Parameter(
                        parameter_config["beta_index"][0],
                        "beta_index",
                        False,
                        parameter_config["beta_index_range"],
                    )
                )
            else:
                for c, idx in enumerate(parameter_config["beta_index"]):
                    name = f"ps_{c}_beta_index"
                    beta.append(
                        Parameter(
                            idx,
                            name,
                            not "beta_index" in parameter_config["fit_params"],
                            parameter_config["beta_index_range"],
                        )
                    )

            if "E0_src" in parameter_config["fit_params"] and share_src_index:
                E0_src.append(
                    Parameter(
                        u.Quantity(parameter_config["E0_src"][0]),
                        "E0_src",
                        False,
                        tuple(
                            u.Quantity(_).to_value(u.GeV)
                            for _ in parameter_config["E0_src_range"]
                        )
                        * u.GeV,
                        ParScale.log,
                    )
                )
            else:
                for c, val in enumerate(parameter_config["E0_src"]):
                    name = f"ps_{c}_E0_src"
                    E0_src.append(
                        Parameter(
                            u.Quantity(val),
                            name,
                            not "E0_src" in parameter_config["fit_params"],
                            tuple(
                                u.Quantity(_).to_value(u.GeV)
                                for _ in parameter_config["E0_src_range"]
                            )
                            * u.GeV,
                            ParScale.log,
                        )
                    )
        if parameter_config["source_type"] == "pgamma" and share_src_index:
            E0_src.append(
                Parameter(
                    u.Quantity(parameter_config["E0_src"][0]),
                    "E0_src",
                    False,
                    tuple(
                        u.Quantity(_).to_value(u.GeV)
                        for _ in parameter_config["E0_src_range"]
                    )
                    * u.GeV,
                    ParScale.log,
                )
            )
        elif parameter_config["source_type"] == "pgamma":
            for c, val in enumerate(parameter_config["E0_src"]):
                name = f"ps_{c}_E0_src"
                E0_src.append(
                    Parameter(
                        u.Quantity(val),
                        name,
                        False,
                        tuple(
                            u.Quantity(_).to_value(u.GeV)
                            for _ in parameter_config["E0_src_range"]
                        )
                        * u.GeV,
                        ParScale.log,
                    )
                )

        if parameter_config.source_type == "SeyfertII" and share_src_index:
            name = "eta"
            eta.append(
                Parameter(
                    parameter_config.eta[0],
                    name,
                    False,
                    parameter_config.eta_range,
                    ParScale.lin,
                )
            )

        elif parameter_config.source_type == "SeyfertII":
            for c, val in enumerate(parameter_config.eta):
                name = f"ps_{c}_eta"
                eta.append(
                    Parameter(
                        val,
                        name,
                        False,
                        parameter_config.eta_range,
                        ParScale.lin,
                    )
                )
        if parameter_config.source_type == "SeyfertII":
            logLx = parameter_config.logLx

        if "Nex_src" in parameter_config["fit_params"]:
            Nex_src = Parameter(
                0.0, "Nex_src", fixed=True, par_range=parameter_config["Nex_src_range"]
            )

        diff_index = Parameter(
            parameter_config["diff_index"],
            "diff_index",
            fixed=False,
            par_range=parameter_config["diff_index_range"],
        )
        L = []
        # P acts as luminosity? I guess
        # reuse share_L for P
        P = []
        if not share_L and not parameter_config.source_type == "SeyfertII":
            for c, Lumi in enumerate(parameter_config["L"]):
                name = f"ps_{c}_luminosity"
                L.append(
                    Parameter(
                        u.Quantity(Lumi),
                        name,
                        True,
                        tuple(
                            u.Quantity(_).to_value(u.GeV / u.s)
                            for _ in parameter_config["L_range"]
                        )
                        * (u.GeV / u.s),
                    )
                )
        elif not parameter_config.source_type == "SeyfertII":
            L.append(
                Parameter(
                    u.Quantity(parameter_config["L"][0]),
                    "luminosity",
                    fixed=True,
                    par_range=tuple(
                        u.Quantity(_).to_value(u.GeV / u.s)
                        for _ in parameter_config["L_range"]
                    )
                    * u.GeV
                    / u.s,
                )
            )
        elif share_L:
            P.append(
                Parameter(
                    parameter_config.P[0],
                    "pressure_ratio",
                    fixed=True,
                    par_range=parameter_config.P_range,
                )
            )
        else:
            for c, pressure in enumerate(parameter_config.P):
                name = f"ps_{c}_pressure_ratio"
                P.append(
                    Parameter(
                        pressure,
                        name,
                        fixed=True,
                        par_range=parameter_config.P_range,
                    )
                )

        diffuse_norm = Parameter(
            u.Quantity(parameter_config["diff_norm"]),
            "diffuse_norm",
            True,
            tuple(
                u.Quantity(_).to_value(1 / u.GeV / u.m**2 / u.s)
                for _ in parameter_config.diff_norm_range
            )
            * (1 / u.GeV / u.m**2 / u.s),
        )
        Enorm = Parameter(u.Quantity(parameter_config["Enorm"]), "Enorm", fixed=True)
        Emin = Parameter(u.Quantity(parameter_config["Emin"]), "Emin", fixed=True)
        Emax = Parameter(u.Quantity(parameter_config["Emax"]), "Emax", fixed=True)

        Emin_src = Parameter(
            u.Quantity(parameter_config["Emin_src"]), "Emin_src", fixed=True
        )
        Emax_src = Parameter(
            u.Quantity(parameter_config["Emax_src"]), "Emax_src", fixed=True
        )

        Emin_diff = Parameter(
            u.Quantity(parameter_config["Emin_diff"]), "Emin_diff", fixed=True
        )
        Emax_diff = Parameter(
            u.Quantity(parameter_config["Emax_diff"]), "Emax_diff", fixed=True
        )

        if parameter_config["Emin_det_eq"]:
            Emin_det = Parameter(
                u.Quantity(parameter_config["Emin_det"]), "Emin_det", fixed=True
            )

        else:
            for dm in Refrigerator.detectors:
                # Create a parameter for each detector
                # If the detector is not used, the parameter is disregarded
                _ = Parameter(
                    u.Quantity(parameter_config[f"Emin_det_{dm.P}"]),
                    f"Emin_det_{dm.P}",
                    fixed=True,
                )

        dec = (
            np.array(
                [u.Quantity(_).to_value(u.deg) for _ in parameter_config["src_dec"]]
            )
            << u.deg
        )
        ra = (
            np.array(
                [u.Quantity(_).to_value(u.deg) for _ in parameter_config["src_ra"]]
            )
            << u.deg
        )
        _frame = parameter_config["frame"]
        if _frame == "detector":
            frame = DetectorFrame
        elif _frame == "source":
            frame = SourceFrame
        else:
            raise ValueError("No other frame implemented")

        sources = Sources()

        for c in range(len(dec)):
            if share_L and not parameter_config.source_type == "SeyfertII":
                Lumi = L[0]
            elif not parameter_config.source_type == "SeyfertII":
                Lumi = L[c]
            elif share_L:
                _P = P[0]
            else:
                _P = P[c]

            if share_src_index:
                if index and "src_index" in parameter_config["fit_params"]:
                    idx = index[0]
                elif parameter_config.source_type == "pgamma":
                    pass
                elif parameter_config.source_type == "SeyfertII":
                    _eta = eta[0]
                else:
                    idx = index[c]
                if beta and "beta_index" in parameter_config["fit_params"]:
                    idx_beta = beta[0]
                elif beta:
                    idx_beta = beta[c]
                if E0_src and "E0_src" in parameter_config["fit_params"]:
                    E0 = E0_src[0]
                elif E0_src:
                    E0 = E0_src[c]
            else:
                if index:
                    idx = index[c]
                if beta:
                    idx_beta = beta[c]
                if E0_src:
                    E0 = E0_src[c]
                if eta:
                    _eta = eta[c]
            if parameter_config.source_type not in ("pgamma", "SeyfertII"):
                args = (
                    f"ps_{c}",
                    dec[c],
                    ra[c],
                    Lumi,
                    idx,
                    parameter_config["z"][c],
                    Emin_src,
                    Emax_src,
                    frame,
                )
            if parameter_config.source_type == "twice-broken-power-law":
                method = PointSource.make_twicebroken_powerlaw_source
            elif parameter_config.source_type == "power-law":
                method = PointSource.make_powerlaw_source
            elif parameter_config.source_type == "logparabola":
                method = PointSource.make_logparabola_source
                args = (
                    f"ps_{c}",
                    dec[c],
                    ra[c],
                    Lumi,
                    idx,
                    idx_beta,
                    parameter_config["z"][c],
                    Emin_src,
                    Emax_src,
                    E0,
                    frame,
                )
            elif parameter_config.source_type == "pgamma":
                method = PointSource.make_pgamma_source
                args = (
                    f"ps_{c}",
                    dec[c],
                    ra[c],
                    Lumi,
                    parameter_config["z"][c],
                    E0,
                    Emin_src,
                    Emax_src,
                    frame,
                )
            elif parameter_config.source_type == "SeyfertII":
                method = PointSource.make_seyfert_source
                args = (
                    f"ps_{c}",
                    dec[c],
                    ra[c],
                    logLx[c],
                    _P,
                    _eta,
                    parameter_config.z[c],
                )
            point_source = method(*args)

            sources.add(point_source)

        if (
            parameter_config.diffuse or parameter_config.diffuse
        ) and parameter_config.data_bg:
            raise ValueError(
                "Cannot combine physical background model with data-driven background model."
            )

        if parameter_config.diffuse:
            sources.add_diffuse_component(
                diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff, 0.0
            )
            # F_diff = Parameter.get_parameter("F_diff")
            # F_diff.par_range = parameter_config.F_diff_range * (1 / u.m**2 / u.s)

        if parameter_config.atmospheric:
            sources.add_atmospheric_component(cache_dir=mceq)
            F_atmo = Parameter.get_parameter("F_atmo")
            F_atmo.par_range = tuple(
                u.Quantity(_).to_value(1 / u.m**2 / u.s)
                for _ in parameter_config.F_atmo_range
            ) * (1 / u.m**2 / u.s)

        if parameter_config.data_bg:
            sources.add_background(*self.detector_model)

        self._sources = sources

        return sources

    @property
    def MJD_min(self):
        return self._hnu_config.parameter_config.MJD_min

    @property
    def MJD_max(self):
        return self._hnu_config.parameter_config.MJD_max

    @property
    def ROI(self):
        ROIList.clear_registry()
        parameter_config = self._hnu_config.parameter_config
        roi_config = self._hnu_config.roi_config
        src_dec = (
            np.array([u.Quantity(_).to_value(u.deg) for _ in parameter_config.src_dec])
            << u.deg
        )
        src_ra = (
            np.array([u.Quantity(_).to_value(u.deg) for _ in parameter_config.src_ra])
            << u.deg
        )

        self.check_units(roi_config.RA, u.deg)
        self.check_units(roi_config.DEC, u.deg)
        RA = u.Quantity(roi_config.RA)
        DEC = u.Quantity(roi_config.DEC)
        if not np.isclose(RA, -1.0 * u.deg) and not np.isclose(DEC, -91.0 * u.deg):
            # Use provided center
            center = SkyCoord(ra=RA, dec=DEC, frame="icrs")
            provided_center = True
        else:
            center = SkyCoord(ra=src_ra, dec=src_dec, frame="icrs")
            provided_center = False

        roi_config = self._hnu_config.roi_config
        self.check_units(roi_config.size, u.deg)
        size = u.Quantity(roi_config.size)
        apply_roi = roi_config.apply_roi

        if apply_roi and len(src_dec) > 1 and not roi_config.roi_type == "CircularROI":
            raise ValueError("Only CircularROIs can be stacked")
        MJD_min = self.MJD_min if not np.isclose(self.MJD_min, 98.0) else 0.0
        MJD_max = self.MJD_max if not np.isclose(self.MJD_max, 100.0) else 99999.0

        self.check_units(roi_config.RA_min, u.deg)
        self.check_units(roi_config.RA_max, u.deg)
        self.check_units(roi_config.DEC_min, u.deg)
        self.check_units(roi_config.DEC_max, u.deg)

        ra_min = u.Quantity(roi_config.RA_min)
        ra_max = u.Quantity(roi_config.RA_max)
        dec_min = u.Quantity(roi_config.DEC_min)
        dec_max = u.Quantity(roi_config.DEC_max)
        if roi_config.roi_type == "CircularROI":
            if not provided_center:
                for c in range(len(src_dec)):
                    CircularROI(
                        center[c],
                        size,
                        apply_roi=apply_roi,
                        MJD_min=MJD_min,
                        MJD_max=MJD_max,
                    )
            else:
                CircularROI(
                    center, size, apply_roi=apply_roi, MJD_min=MJD_min, MJD_max=MJD_max
                )
        elif roi_config.roi_type == "RectangularROI":
            size = size.to(u.rad)
            if (
                not (
                    np.isclose(ra_min, -1.0 * u.deg)
                    and np.isclose(ra_max, 361.0 * u.deg)
                    and np.isclose(dec_min, -91.0 * u.deg)
                    and np.isclose(dec_max, 91.0 * u.deg)
                )
                and not provided_center
            ):
                RectangularROI(
                    RA_min=ra_min,
                    RA_max=ra_max,
                    DEC_min=dec_min,
                    DEC_max=dec_max,
                    MJD_min=MJD_min,
                    MJD_max=MJD_max,
                    apply_roi=apply_roi,
                )
            elif not provided_center:
                RectangularROI(
                    RA_min=src_ra[0] - size,
                    RA_max=src_ra[0] + size,
                    DEC_min=src_dec[0] - size,
                    DEC_max=src_dec[0] + size,
                    MJD_min=MJD_min,
                    MJD_max=MJD_max,
                    apply_roi=apply_roi,
                )
            else:
                RectangularROI(
                    RA_min=RA - size,
                    RA_max=RA + size,
                    DEC_min=DEC - size,
                    DEC_max=DEC + size,
                    MJD_min=MJD_min,
                    MJD_max=MJD_max,
                    apply_roi=apply_roi,
                )
        elif roi_config.roi_type == "FullSkyROI":
            FullSkyROI(
                MJD_min=MJD_min,
                MJD_max=MJD_max,
            )
        elif roi_config.roi_type == "NorthernSkyROI":
            NorthernSkyROI(MJD_min=MJD_min, MJD_max=MJD_max, apply_roi=apply_roi)

        return ROIList.STACK

    def _is_dm_list(self):
        mjd_min = self.MJD_min
        mjd_max = self.MJD_max
        if not np.isclose(mjd_min, 98.0) and not np.isclose(mjd_max, 100.0):
            return 0
        else:
            return 1

    @property
    def detector_model(self):
        return list(self.obs_time.keys())

    @property
    def obs_time(self):

        if self._is_dm_list():
            dm_keys = [
                Refrigerator.python2dm(_)
                for _ in self._hnu_config.parameter_config.detector_model_type
            ]
            obs_time = self._hnu_config.parameter_config.obs_time
            if obs_time == ["season"]:
                lifetime = LifeTime()
                obs_time = lifetime.lifetime_from_dm(*dm_keys)
                return obs_time
            else:
                return self._get_obs_time_from_config(dm_keys, obs_time)
        else:
            lifetime = LifeTime()
            # check if parameter_config.detector_model_type should be used
            # through parameter_config.restrict_to_list being True
            _time = lifetime.lifetime_from_mjd(self.MJD_min, self.MJD_max)
            if self._hnu_config.parameter_config.restrict_to_list:
                dms = self._hnu_config.parameter_config.detector_model_type
                time = {}
                for dm in dms:
                    dm = Refrigerator.python2dm(dm)
                    try:
                        time[dm] = _time[dm]
                    except KeyError:
                        continue
                _time = time
                if not _time.keys():
                    raise ValueError(
                        "Empty dm list, change MJD or dm selection to sensible values."
                    )
            return _time

    @property
    def events(self):
        _events = Events.from_ev_file(
            *self.detector_model,
            scramble_ra=self._hnu_config.parameter_config.scramble_ra,
        )
        return _events

    @property
    def stan_kwargs(self):
        return self._hnu_config.stan_config

    def create_simulation(self, sources, detector_models, obs_time):

        from hierarchical_nu.simulation import Simulation

        asimov = self._hnu_config.parameter_config.asimov
        sim = Simulation(sources, detector_models, obs_time, asimov=asimov)
        return sim

    def create_fit(self, sources, events, detector_models, obs_time):

        use_event_tag = self._hnu_config.parameter_config.use_event_tag
        from hierarchical_nu.fit import StanFit

        priors = self.priors

        nshards = self._hnu_config.stan_config.threads_per_chain
        fit = StanFit(
            sources,
            detector_models,
            events,
            obs_time,
            priors=priors,
            nshards=nshards,
            use_event_tag=use_event_tag,
        )
        return fit

    @property
    def priors(self):
        """
        Make priors from config file.
        Assumes default units specified in `hierarchical_nu.priors` for each quantity.
        """
        priors = Priors()
        prior_config = self._hnu_config.prior_config

        def _make_prior(
            multiparameterprior,
            parameterprior,
            prior,
            mu,
            sigma,
            mu_unit: bool,
            sigma_unit: bool,
        ):
            if not isinstance(mu, omegaconf.listconfig.ListConfig) and not isinstance(
                mu, list
            ):
                mu = [mu]
            if not isinstance(
                sigma, omegaconf.listconfig.ListConfig
            ) and not isinstance(sigma, list):
                sigma = [sigma]

            if mu_unit:
                mu = [u.Quantity(_) for _ in mu]
            if sigma_unit:
                sigma = [u.Quantity(_) for _ in sigma]
            if len(mu) > 1 and len(sigma) > 1:
                return multiparameterprior(
                    [parameterprior(prior, mu=m, sigma=s) for m, s in zip(mu, sigma)]
                )
            elif len(mu) > 1:
                return multiparameterprior(
                    [parameterprior(prior, mu=m, sigma=sigma[0]) for m in mu]
                )
            elif len(sigma) > 1:
                return multiparameterprior(
                    [parameterprior(prior, mu=mu[0], sigma=s) for s in sigma]
                )
            else:
                return parameterprior(prior, mu=mu[0], sigma=sigma[0])

        for p, vals in prior_config.items():
            if vals.name == "NormalPrior":
                prior = NormalPrior
                mu = vals.mu
                sigma = vals.sigma
                sigma_unit = True
            elif vals.name == "LogNormalPrior":
                prior = LogNormalPrior
                mu = vals.mu
                sigma = vals.sigma
                sigma_unit = False
            elif vals.name == "ParetoPrior":
                prior = ParetoPrior
                xmin = vals.xmin
                alpha = vals.alpha
            elif vals.name == "Ignorance":
                prior = Ignorance
            else:
                raise NotImplementedError("Prior type not recognised.")

            if p == "src_index":
                if prior != Ignorance:
                    self.check_units(mu, 1)
                    self.check_units(sigma, 1)
                else:
                    mu = 1.0
                    sigma = 1.0
                priors.src_index = _make_prior(
                    MultiSourceIndexPrior, IndexPrior, prior, mu, sigma, False, False
                )
            elif p == "beta_index":
                self.check_units(mu, 1)
                self.check_units(sigma, 1)
                priors.beta_index = _make_prior(
                    MultiSourceIndexPrior, IndexPrior, prior, mu, sigma, False, False
                )
            elif p == "E0_src":
                self.check_units(mu, u.GeV)
                if prior == NormalPrior:
                    self.check_units(sigma, u.GeV)
                elif prior == LogNormalPrior:
                    self.check_units(sigma, 1)
                else:
                    raise NotImplementedError("Prior not recognised for E0_src.")
                priors.E0_src = _make_prior(
                    MultiSourceEnergyPrior,
                    EnergyPrior,
                    prior,
                    mu,
                    sigma,
                    True,
                    sigma_unit,
                )
            elif p == "eta":
                if prior != Ignorance:
                    self.check_units(mu, 1)
                    self.check_units(sigma, 1)
                else:
                    mu = 1.0
                    sigma = 1.0
                prior.eta = _make_prior(
                    MultiSourceEtaPrior, EtaPrior, prior, mu, sigma, False, False
                )
            elif p == "P":
                self.check_units(mu, 1)
                self.check_units(sigma, 1)
                priors.pressure_ratio = _make_prior(
                    MultiSourcePressureRatioPrior,
                    PressureRatioPrior,
                    prior,
                    mu,
                    sigma,
                    False,
                    False,
                )
            elif p == "L":
                if prior == ParetoPrior:
                    self.check_units(xmin, u.GeV / u.s)
                    self.check_units(alpha, 1)
                    priors.luminosity = LuminosityPrior(
                        prior, xmin=u.Quantity(xmin), alpha=alpha
                    )
                    continue

                self.check_units(mu, u.GeV / u.s)
                if prior == NormalPrior:
                    self.check_units(sigma, u.GeV / u.s)
                elif prior == LogNormalPrior:
                    self.check_units(sigma, 1)
                else:
                    raise NotImplementedError("Prior not recognised for E0_src.")
                priors.luminosity = _make_prior(
                    MultiSourceLuminosityPrior,
                    LuminosityPrior,
                    prior,
                    mu,
                    sigma,
                    True,
                    sigma_unit,
                )

            elif p == "diff_index":
                priors.diff_index = IndexPrior(prior, mu=mu, sigma=sigma)

            elif p == "diff_flux":
                self.check_units(mu, 1 / u.GeV / u.m**2 / u.s)
                if prior == NormalPrior:
                    self.check_units(sigma, 1 / u.GeV / u.m**2 / u.s)
                    priors.diffuse_flux = DifferentialFluxPrior(
                        prior,
                        mu=u.Quantity(mu),
                        sigma=u.Quantity(sigma),
                    )
                elif prior == LogNormalPrior:
                    self.check_units(sigma, 1)
                    priors.diffuse_flux = DifferentialFluxPrior(
                        prior, mu=u.Quantity(mu), sigma=sigma
                    )
                else:
                    raise NotImplementedError("Prior not recognised.")
            elif p == "atmo_flux":
                self.check_units(mu, 1 / u.m**2 / u.s)
                if prior == NormalPrior:
                    self.check_units(sigma, 1 / u.m**2 / u.s)
                    priors.atmospheric_flux = FluxPrior(
                        prior, mu=u.Quantity(mu), sigma=u.Quantity(sigma)
                    )
                elif prior == LogNormalPrior:
                    self.check_units(sigma, 1)
                    priors.atmospheric_flux = FluxPrior(
                        prior, mu=u.Quantity(mu), sigma=sigma
                    )
                else:
                    raise NotImplementedError("Prior not recognised.")
            elif p == "Nex_src":
                self.check_units(mu, 1)
                self.check_units(sigma, 1)
                priors.Nex_src = NexPrior(
                    prior, mu=mu, sigma=sigma
                )

        return priors

    @classmethod
    def _get_dm_from_config(cls, dm_key):
        return [Refrigerator.python2dm(dm) for dm in dm_key]

    @classmethod
    def _get_obs_time_from_config(cls, dms, obs_time):
        cls.check_units(obs_time, u.yr)
        return {dm: u.Quantity(obs_time[c]) for c, dm in enumerate(dms)}
