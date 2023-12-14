import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import ligo.skymap.plot


from icecube_tools.utils.vMF import get_kappa

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.utils.roi import (
    ROI,
    RectangularROI,
    CircularROI,
    FullSkyROI,
    ROIList,
)
from hierarchical_nu.utils.plotting import SphericalCircle
from hierarchical_nu.detector.icecube import Refrigerator
from hierarchical_nu.detector.icecube import EventType

import logging

from typing import List
import numpy.typing as npt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Events:
    """
    Events class for the storage of event observables
    """

    @u.quantity_input
    def __init__(
        self,
        energies: u.GeV,
        coords: SkyCoord,
        types,
        ang_errs: u.deg,
        mjd: Time,
    ):
        """
        Events class for the storage of event observables
        """

        self._recognised_types = [_.S for _ in Refrigerator.detectors]

        self._energies = energies

        self._mjd = mjd

        coords.representation_type = "spherical"
        self._coords = coords
        coords.representation_type = "cartesian"
        self._unit_vectors = np.array(
            [coords.x.value, coords.y.value, coords.z.value]
        ).T

        if all([t in self._recognised_types for t in types]):
            self._types = types
        else:
            raise ValueError("Event types not recognised")

        self._ang_errs = ang_errs

    def remove(self, i):
        """
        Remove the event at index i
        :param i: Event index
        """
        self._energies = np.delete(self._energies, i)
        self._coords = np.delete(self._coords, i)
        self._unit_vectors = np.delete(self._unit_vectors, i, axis=0)
        self._types = np.delete(self._types, i)
        self._ang_errs = np.delete(self._ang_errs, i)
        self._mjd = np.delete(self._mjd, i)

    @property
    def N(self):
        try:
            return self.types.size
        except AttributeError:
            return len(self.types)

    def select(self, mask: npt.NDArray[np.bool_]):
        """
        Select some subset of existing events by providing a mask.
        :param mask: Array of bools with same length as event properties.
        """

        assert len(mask) == self.N

        self._energies = self._energies[mask]
        self._coords = self._coords[mask]
        self._unit_vectors = self._unit_vectors[mask]
        self._types = self._types[mask]
        self._ang_errs = self._ang_errs[mask]
        self._mjd = self._mjd[mask]

    @property
    def energies(self):
        return self._energies

    @property
    def coords(self):
        return self._coords

    @property
    def unit_vectors(self):
        return self._unit_vectors

    @property
    def types(self):
        return self._types

    @property
    def ang_errs(self):
        return self._ang_errs

    @property
    def kappas(self):
        return get_kappa(self._ang_errs.to_value(u.deg), p=0.683)

    @property
    def mjd(self):
        return self._mjd

    @classmethod
    def from_file(cls, filename, group_name=None):
        with h5py.File(filename, "r") as f:
            if group_name is None:
                events_folder = f["events"]
            else:
                events_folder = f[group_name]

            energies = events_folder["energies"][()] * u.GeV
            uvs = events_folder["unit_vectors"][()]
            types = events_folder["event_types"][()]
            ang_errs = events_folder["ang_errs"][()] * u.deg

            # For backwards compatibility
            try:
                mjd = events_folder["mjd"][()]
            except KeyError:
                mjd = [99.0] * len(energies)

        coords = SkyCoord(
            uvs.T[0], uvs.T[1], uvs.T[2], representation_type="cartesian", frame="icrs"
        )

        time = Time(mjd, format="mjd")

        coords.representation_type = "spherical"

        ra = coords.ra.rad * u.rad
        dec = coords.dec.rad * u.rad

        coords.representation_type = "cartesian"
        mask = []
        if ROIList.STACK:
            logger.info("Applying ROIs to event selection")
            for roi in ROIList.STACK:
                # TODO add reco energy cut for all event types
                if isinstance(roi, CircularROI):
                    mask.append(
                        (roi.radius >= roi.center.separation(coords))
                        & (mjd <= roi.MJD_max)
                        & (mjd >= roi.MJD_min)
                    )
                else:
                    if roi.RA_min > roi.RA_max:
                        mask.append(
                            (dec <= roi.DEC_max)
                            & (dec >= roi.DEC_min)
                            & ((ra >= roi.RA_min) | (ra <= roi.RA_max))
                            & (mjd <= roi.MJD_max)
                            & (mjd >= roi.MJD_min)
                        )

                    else:
                        mask.append(
                            (dec <= roi.DEC_max)
                            & (dec >= roi.DEC_min)
                            & (ra >= roi.RA_min)
                            & (ra <= roi.RA_max)
                            & (mjd <= roi.MJD_max)
                            & (mjd >= roi.MJD_min)
                        )

            idxs = np.logical_or.reduce(mask)

            return cls(
                energies[idxs], coords[idxs], types[idxs], ang_errs[idxs], time[idxs]
            )
        else:
            logger.info("Applying no ROIs to event selection")
            return cls(energies, coords, types, ang_errs, time)

    def to_file(self, filename, append=False, group_name=None):
        self._file_keys = ["energies", "unit_vectors", "event_types", "ang_errs", "mjd"]
        self._file_values = [
            self.energies.to(u.GeV).value,
            self.unit_vectors,
            self.types,
            self.ang_errs.to(u.deg).value,
            self.mjd.mjd,
        ]

        if append:
            with h5py.File(filename, "r+") as f:
                if group_name is None:
                    event_folder = f.create_group("events")
                else:
                    event_folder = f.create_group(group_name)

                for key, value in zip(self._file_keys, self._file_values):
                    event_folder.create_dataset(key, data=value)

        else:
            with h5py.File(filename, "w") as f:
                event_folder = f.create_group("events")

                for key, value in zip(self._file_keys, self._file_values):
                    event_folder.create_dataset(key, data=value)

    @classmethod
    def from_ev_file(cls, *seasons: EventType):
        """
        Load events from the 2021 data release
        :param seasons: arbitrary number of `EventType` identifying detector seasons of r2021 release.
        :return: :class:`hierarchical_nu.events.Events`
        """

        from icecube_tools.utils.data import RealEvents

        # Borrow from icecube_tools
        # Already exclude low energy events here, would be quite difficult later on
        try:
            _Emin_det = Parameter.get_parameter("Emin_det").value.to_value(u.GeV)
            events = RealEvents.from_event_files(*(s.P for s in seasons), use_all=True)
            events.restrict(ereco_low=_Emin_det)
        except ValueError:
            events = RealEvents.from_event_files(*(s.P for s in seasons), use_all=True)
            # Create a dict of masks for each season
            mask = {}
            for s in seasons:
                try:
                    _Emin_det = Parameter.get_parameter(
                        f"Emin_det_{s.P}"
                    ).value.to_value(u.GeV)
                    mask[s.P] = events.reco_energy[s.P] >= _Emin_det
                except ValueError:
                    raise ValueError("Emin_det not defined for all seasons.")
            events.mask = mask

        ra = np.hstack([events.ra[s.P] * u.rad for s in seasons])
        dec = np.hstack([events.dec[s.P] * u.rad for s in seasons])
        reco_energy = np.hstack([events.reco_energy[s.P] * u.GeV for s in seasons])
        types = np.hstack([events.ra[s.P].size * [s.S] for s in seasons])
        mjd = np.hstack([events.mjd[s.P] for s in seasons])

        # Conversion from 50% containment to 68% is already done in RealEvents
        ang_err = np.hstack([events.ang_err[s.P] * u.deg for s in seasons])
        coords = SkyCoord(ra=ra, dec=dec, frame="icrs")

        mask = []
        for roi in ROIList.STACK:
            if isinstance(roi, CircularROI):
                mask.append(
                    (
                        (roi.radius >= roi.center.separation(coords))
                        & (mjd <= roi.MJD_max)
                        & (mjd >= roi.MJD_min)
                    )
                )
            else:
                if roi.RA_min > roi.RA_max:
                    mask.append(
                        (dec <= roi.DEC_max)
                        & (dec >= roi.DEC_min)
                        & ((ra >= roi.RA_min) | (ra <= roi.RA_max))
                        & (mjd <= roi.MJD_max)
                        & (mjd >= roi.MJD_min)
                    )

                else:
                    mask.append(
                        (dec <= roi.DEC_max)
                        & (dec >= roi.DEC_min)
                        & (ra >= roi.RA_min)
                        & (ra <= roi.RA_max)
                        & (mjd <= roi.MJD_max)
                        & (mjd >= roi.MJD_min)
                    )

        mjd = Time(mjd, format="mjd")

        idxs = np.logical_or.reduce(mask)

        return cls(
            reco_energy[idxs], coords[idxs], types[idxs], ang_err[idxs], mjd[idxs]
        )

    @u.quantity_input
    def plot_energy(self, center_coords: SkyCoord, radius: 3 * u.deg, lw: float = 1.0):
        fig, ax = plt.subplots(
            subplot_kw={
                "projection": "astro degrees zoom",
                "center": center_coords,
                "radius": f"{radius.to_value(u.deg)} deg",
            },
            dpi=150,
        )

        logNorm = colors.LogNorm(
            self.energies.to_value(u.GeV).min(),
            self.energies.to_value(u.GeV).max(),
            clip=True,
        )
        linNorm = colors.Normalize(
            self.energies.to_value(u.GeV).min(),
            self.energies.to_value(u.GeV).max(),
            clip=True,
        )

        mapper = cm.ScalarMappable(norm=logNorm, cmap=cm.viridis_r)
        color = mapper.to_rgba(self.energies.to_value(u.GeV))

        self.coords.representation_type = "spherical"
        for r, d, c, e, energy in zip(
            self.coords.icrs.ra,
            self.coords.icrs.dec,
            color,
            self.ang_errs,
            np.log10(self.energies.to_value(u.GeV)),
        ):
            circle = SphericalCircle(
                (r, d),
                e,
                edgecolor=c,
                alpha=0.5,
                transform=ax.get_transform("icrs"),
                zorder=linNorm(energy) + 1,
                facecolor="None",
                lw=lw,
            )

            ax.add_patch(circle)

        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")

        fig.colorbar(mapper, ax=ax, label=r"$\hat E~[\mathrm{GeV}]$")

        return fig, ax

    @u.quantity_input
    def plot_radial_excess(self, center: SkyCoord, radius: u.deg = 5 * u.deg):
        """
        Plot histogram of radial distance to a source located at center.
        Bin edges are equdistant in angle squared such that equal areas in polar coordinates
        (assuming Euclidian space for small angles) are covered by each bin.
        :param center: SkyCoord of center
        :param radius: Max radius of histogram
        """

        r2_bins = np.arange(
            0.0, np.power(radius.to_value(u.deg), 2) + 1.0 / 3.0, 1.0 / 3.0
        )
        sep = center.separation(self.coords).deg

        fig, ax = plt.subplots()
        ax.hist(sep**2, r2_bins, histtype="step")
        ax.set_xlabel("$\Psi^2$ [deg$^2$]")
        ax.set_ylabel("counts")

        return fig, ax
