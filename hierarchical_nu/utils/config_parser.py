from hierarchical_nu.priors import (
    Priors,
    LogNormalPrior,
    NormalPrior,
    # LuminosityPrior,
    # IndexPrior,
    # FluxPrior,
)
from hierarchical_nu.utils.config import hnu_config, HierarchicalNuConfig
from hierarchical_nu.source.source import Sources, PointSource
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
from hierarchical_nu.events import Events

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np


class ConfigParser:
    def __init__(self, hnu_config: HierarchicalNuConfig):
        self._hnu_config = hnu_config

    def sources(self):
        sources = Sources()
        parameter_config = self._hnu_config["parameter_config"]
        share_L = parameter_config["share_L"]
        share_src_index = parameter_config["share_src_index"]

        Parameter.clear_registry()
        indices = []
        if not share_src_index:
            for c, idx in enumerate(parameter_config["src_index"]):
                name = f"ps_{c}_src_index"
                indices.append(
                    Parameter(
                        idx,
                        name,
                        fixed=False,
                        par_range=parameter_config["src_index_range"],
                    )
                )
        else:
            indices.append(
                Parameter(
                    parameter_config["src_index"][0],
                    "src_index",
                    fixed=False,
                    par_range=parameter_config["src_index_range"],
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

        sources = Sources()

        for c in range(len(dec)):
            if share_L:
                Lumi = L[0]
            else:
                Lumi = L[c]

            if share_src_index:
                idx = indices[0]
            else:
                idx = indices[c]
            point_source = PointSource.make_powerlaw_source(
                f"ps_{c}",
                dec[c],
                ra[c],
                Lumi,
                idx,
                parameter_config["z"][c],
                Emin_src,
                Emax_src,
            )

            sources.add(point_source)
        if parameter_config.diffuse:
            sources.add_diffuse_component(
                diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff, 0.0
            )
        if parameter_config.atmospheric:
            sources.add_atmospheric_component()

        return sources

    def ROI(self):
        ROIList.clear_registry()
        parameter_config = self._hnu_config.parameter_config
        roi_config = hnu_config.roi_config
        dec = np.deg2rad(parameter_config.src_dec) * u.rad
        ra = np.deg2rad(parameter_config.src_ra) * u.rad
        center = SkyCoord(ra=ra, dec=dec, frame="icrs")

        roi_config = hnu_config.roi_config
        size = roi_config.size * u.deg
        apply_roi = roi_config.apply_roi

        if apply_roi and len(dec) > 1 and not roi_config.roi_type == "CircularROI":
            raise ValueError("Only CircularROIs can be stacked")
        if roi_config.roi_type == "CircularROI":
            for c in range(len(dec)):
                CircularROI(center[c], size, apply_roi=apply_roi)
        elif roi_config.roi_type == "RectangularROI":
            size = size.to(u.rad)
            RectangularROI(
                RA_min=ra[0] - size,
                RA_max=ra[0] + size,
                DEC_min=dec[0] - size,
                DEC_max=dec[0] + size,
                apply_roi=apply_roi,
            )
        elif roi_config.roi_type == "FullSkyROI":
            FullSkyROI()
        elif roi_config.roi_type == "NorthernSkyROI":
            NorthernSkyROI()

    def detector_model(self):
        dm_keys = self._hnu_config.parameter_config.detector_model_type
        return self._get_dm_from_config(dm_keys)

    def obs_time(self):
        dm_keys = self.detector_model()
        obs_time = self._hnu_config.parameter_config.obs_time
        return self._get_obs_time_from_config(dm_keys, obs_time)

    def simulation(self, sources, detector_models, obs_time):
        asimov = self._hnu_config.parameter_config.asimov
        sim = Simulation(sources, detector_models, obs_time, asimov=asimov)
        return sim

    def fit(self, sources, events, detector_models, obs_time):
        priors = self.priors()

        nshards = self._hnu_config.parameter_config.threads_per_chain
        fit = StanFit(sources, detector_models, events, obs_time, priors, nshards)
        return fit

    def priors(self):
        prior_config = self._hnu_config.prior_config
        priors = Priors()
        priors.luminosity = self._make_prior(prior_config["L"])
        priors.src_index = self._make_prior(prior_config["src_index"])
        priors.atmospheric_flux = self._make_prior(prior_config["atmo_flux"])
        priors.diffuse_flux = self._make_prior(prior_config["diff_flux"])
        priors.diff_index = self._make_prior(prior_config["diff_index"])
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
