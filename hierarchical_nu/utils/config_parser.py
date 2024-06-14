from hierarchical_nu.priors import (
    Priors,
    LogNormalPrior,
    NormalPrior,
    ParetoPrior,
    LuminosityPrior,
    IndexPrior,
    FluxPrior,
)
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.source.source import Sources, PointSource, SourceFrame, DetectorFrame
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.detector.icecube import Refrigerator
from hierarchical_nu.utils.roi import (
    ROIList,
    CircularROI,
    NorthernSkyROI,
    FullSkyROI,
    RectangularROI,
)
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.detector.input import mceq
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.events import Events

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np


class ConfigParser:
    def __init__(self, hnu_config: HierarchicalNuConfig):
        self._hnu_config = hnu_config

    @property
    def sources(self):

        parameter_config = self._hnu_config["parameter_config"]
        share_L = parameter_config["share_L"]
        share_src_index = parameter_config["share_src_index"]

        Parameter.clear_registry()
        index = []
        beta = []
        E0_src = []
        if parameter_config["source_type"] == "power-law" or \
        parameter_config["source_type"] == "twice-broken-power-law" or \
        parameter_config["source_type"] == "logparabola":
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
                        parameter_config["E0_src"][0] * u.GeV,
                        "E0_src",
                        False,
                        parameter_config["E0_src_range"] * u.GeV,
                    )
                )
            else:
                for c, idx in enumerate(parameter_config["E0_src"]):
                    name = f"ps_{c}_E0_src"
                    E0_src.append(
                        Parameter(
                            idx * u.GeV,
                            name,
                            not "E0_src" in parameter_config["fit_params"],
                            parameter_config["E0_src_range"] * u.GeV,
                        )
                    )

        diff_index = Parameter(
            parameter_config["diff_index"],
            "diff_index",
            fixed=False,
            par_range=parameter_config["diff_index_range"],
        )
        L = []
        if not share_L:
            for c, Lumi in enumerate(parameter_config["L"]):
                name = f"ps_{c}_luminosity"
                L.append(
                    Parameter(
                        Lumi * u.erg / u.s,
                        name,
                        fixed=True,
                        par_range=parameter_config["L_range"] * u.erg / u.s,
                    )
                )
        else:
            L.append(
                Parameter(
                    parameter_config["L"][0] * u.erg / u.s,
                    "luminosity",
                    fixed=False,
                    par_range=parameter_config["L_range"] * u.erg / u.s,
                )
            )
        diffuse_norm = Parameter(
            parameter_config["diff_norm"] * 1 / (u.GeV * u.m**2 * u.s),
            "diffuse_norm",
            fixed=True,
            par_range=(0, np.inf),
        )
        Enorm = Parameter(parameter_config["Enorm"] * u.GeV, "Enorm", fixed=True)
        Emin = Parameter(parameter_config["Emin"] * u.GeV, "Emin", fixed=True)
        Emax = Parameter(parameter_config["Emax"] * u.GeV, "Emax", fixed=True)

        Emin_src = Parameter(
            parameter_config["Emin_src"] * u.GeV, "Emin_src", fixed=True
        )
        Emax_src = Parameter(
            parameter_config["Emax_src"] * u.GeV, "Emax_src", fixed=True
        )

        Emin_diff = Parameter(
            parameter_config["Emin_diff"] * u.GeV, "Emin_diff", fixed=True
        )
        Emax_diff = Parameter(
            parameter_config["Emax_diff"] * u.GeV, "Emax_diff", fixed=True
        )

        if parameter_config["Emin_det_eq"]:
            Emin_det = Parameter(
                parameter_config["Emin_det"] * u.GeV, "Emin_det", fixed=True
            )

        else:
            for dm in Refrigerator.detectors:
                # Create a parameter for each detector
                # If the detector is not used, the parameter is disregarded
                _ = Parameter(
                    parameter_config[f"Emin_det_{dm.P}"] * u.GeV,
                    f"Emin_det_{dm.P}",
                    fixed=True,
                )

        dec = np.deg2rad(parameter_config["src_dec"]) * u.rad
        ra = np.deg2rad(parameter_config["src_ra"]) * u.rad
        _frame = parameter_config["frame"]
        if _frame == "detector":
            frame = DetectorFrame
        elif _frame == "source":
            frame = SourceFrame
        else:
            raise ValueError("No other frame implemented")

        sources = Sources()

        for c in range(len(dec)):
            if share_L:
                Lumi = L[0]
            else:
                Lumi = L[c]

            if share_src_index:
                if index and "src_index" in parameter_config["fit_params"]:
                    idx = index[0]
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
                idx = index[c]
                if beta:
                    idx_beta = beta[c]
                if E0_src:
                    E0 = E0_src[c]
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
            point_source = method(*args)

            sources.add(point_source)

        if parameter_config.diffuse:
            sources.add_diffuse_component(
                diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff, 0.0
            )
            F_diff = Parameter.get_parameter("F_diff")
            F_diff.par_range = parameter_config.F_diff_range * (1 / u.m**2 / u.s)

        if parameter_config.atmospheric:
            sources.add_atmospheric_component(cache_dir=mceq)
            F_atmo = Parameter.get_parameter("F_atmo")
            F_atmo.par_range = parameter_config.F_atmo_range * (1 / u.m**2 / u.s)


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
        dec = np.deg2rad(parameter_config.src_dec) * u.rad
        ra = np.deg2rad(parameter_config.src_ra) * u.rad
        center = SkyCoord(ra=ra, dec=dec, frame="icrs")

        roi_config = self._hnu_config.roi_config
        size = roi_config.size * u.deg
        apply_roi = roi_config.apply_roi

        if apply_roi and len(dec) > 1 and not roi_config.roi_type == "CircularROI":
            raise ValueError("Only CircularROIs can be stacked")
        MJD_min = self.MJD_min if not np.isclose(self.MJD_min, 98.0) else 0.0
        MJD_max = self.MJD_max if not np.isclose(self.MJD_max, 100.0) else 99999.0
        if roi_config.roi_type == "CircularROI":
            for c in range(len(dec)):
                CircularROI(
                    center[c],
                    size,
                    apply_roi=apply_roi,
                    MJD_min=MJD_min,
                    MJD_max=MJD_max,
                )
        elif roi_config.roi_type == "RectangularROI":
            size = size.to(u.rad)
            RectangularROI(
                RA_min=ra[0] - size,
                RA_max=ra[0] + size,
                DEC_min=dec[0] - size,
                DEC_max=dec[0] + size,
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

    def create_simulation(self, sources, detector_models, obs_time):
        asimov = self._hnu_config.parameter_config.asimov
        sim = Simulation(sources, detector_models, obs_time, asimov=asimov)
        return sim

    def create_fit(self, sources, events, detector_models, obs_time):
        priors = self.priors

        nshards = self._hnu_config.parameter_config.threads_per_chain
        fit = StanFit(
            sources,
            detector_models,
            events,
            obs_time,
            priors=priors,
            nshards=nshards,
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

        for p, vals in prior_config.items():
            if vals.name == "NormalPrior":
                prior = NormalPrior
                mu = vals.mu
                sigma = vals.sigma
            elif vals.name == "LogNormalPrior":
                prior = LogNormalPrior
                mu = vals.mu
                sigma = vals.sigma
            elif vals.name == "ParetoPrior":
                prior = ParetoPrior
            else:
                raise NotImplementedError("Prior type not recognised.")

            if p == "src_index":
                priors.src_index = IndexPrior(prior, mu=mu, sigma=sigma)
            elif p == "diff_index":
                priors.diff_index = IndexPrior(prior, mu=mu, sigma=sigma)
            elif p == "L":
                if prior == NormalPrior:
                    priors.luminosity = LuminosityPrior(
                        prior,
                        mu=mu * LuminosityPrior.UNITS,
                        sigma=sigma * LuminosityPrior.UNITS,
                    )
                elif prior == LogNormalPrior:
                    priors.luminosity = LuminosityPrior(
                        prior, mu=mu * LuminosityPrior.UNITS, sigma=sigma
                    )
                elif prior == ParetoPrior:
                    raise NotImplementedError("Prior not recognised.")
            elif p == "diff_flux":
                if prior == NormalPrior:
                    priors.diffuse_flux = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma * FluxPrior.UNITS
                    )
                elif prior == LogNormalPrior:
                    priors.diffuse_flux = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma
                    )
                else:
                    raise NotImplementedError("Prior not recognised.")

            elif p == "atmo_flux":
                if prior == NormalPrior:
                    priors.atmospheric_flux = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma * FluxPrior.UNITS
                    )
                elif prior == LogNormalPrior:
                    priors.atmospheric_flux = FluxPrior(
                        prior, mu=mu * FluxPrior.UNITS, sigma=sigma
                    )
                else:
                    raise NotImplementedError("Prior not recognised.")

        return priors

    @staticmethod
    def _get_dm_from_config(dm_key):
        return [Refrigerator.python2dm(dm) for dm in dm_key]

    @staticmethod
    def _get_obs_time_from_config(dms, obs_time):
        return {dm: obs_time[c] * u.year for c, dm in enumerate(dms)}

    @staticmethod
    def _make_prior(p):
        if p["name"] == "LogNormalPrior":
            prior = LogNormalPrior(mu=np.log(p["mu"]), sigma=p["sigma"])
        elif p["name"] == "NormalPrior":
            prior = NormalPrior(mu=p["mu"], sigma=p["sigma"])
        else:
            raise ValueError("Currently no other prior implemented")
        return prior
