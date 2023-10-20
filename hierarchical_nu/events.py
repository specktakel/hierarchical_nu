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
from hierarchical_nu.utils.roi import ROI, RectangularROI, CircularROI
from hierarchical_nu.utils.plotting import SphericalCircle
from hierarchical_nu.detector.icecube import Refrigerator

import logging

from typing import List

logger = logging.getLogger(__name__)


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

        self.N = len(energies)

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
        self._energies = np.delete(self._energies, i)
        self._coords = np.delete(self._coords, i)
        self._unit_vectors = np.delete(self._unit_vectors, i, axis=0)
        self._types = np.delete(self._types, i)
        self._ang_errs = np.delete(self._ang_errs, i)
        self._mjd = np.delete(self._mjd, i)
        self.N -= 1

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
    def from_file(cls, filename):
        with h5py.File(filename, "r") as f:
            events_folder = f["events"]

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

        try:
            roi = ROI.STACK[0]
        except IndexError:
            roi = RectangularROI()

        # TODO add reco energy cut for all event types
        if roi.RA_min > roi.RA_max:
            mask = np.nonzero(
                (
                    (dec <= roi.DEC_max)
                    & (dec >= roi.DEC_min)
                    & ((ra >= roi.RA_min) | (ra <= roi.RA_max))
                )
            )
        else:
            mask = np.nonzero(
                (
                    (dec <= roi.DEC_max)
                    & (dec >= roi.DEC_min)
                    & (ra >= roi.RA_min)
                    & (ra <= roi.RA_max)
                )
            )

        return cls(
            energies[mask], coords[mask], types[mask], ang_errs[mask], time[mask]
        )

    def to_file(self, filename, append=False):
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
                event_folder = f.create_group("events")

                for key, value in zip(self._file_keys, self._file_values):
                    event_folder.create_dataset(key, data=value)

        else:
            with h5py.File(filename, "w") as f:
                event_folder = f.create_group("events")

                for key, value in zip(self._file_keys, self._file_values):
                    event_folder.create_dataset(key, data=value)

    @classmethod
    def from_ev_file(
        cls,
        *seasons: str,
        **kwargs,
    ):
        """
        Load events from the 2021 data release
        :param seasons: arbitrary number of strings identifying detector seasons of r2021 release.
        :param kwargs: kwargs passed to make an event selection, see `icecube_tools` documentation for details
        :return: :class:`hierarchical_nu.events.Events`
        """

        from icecube_tools.utils.data import RealEvents

        # Borrow from icecube_tools
        use_all = kwargs.pop("use_all", True)
        events = RealEvents.from_event_files(*(s.P for s in seasons), use_all=use_all)
        # Emin_det = {}
        try:
            _Emin_det = Parameter.get_parameter("Emin_det").value.to(u.GeV)
            for s in seasons:
                Emin_det = _Emin_det
        except ValueError:
            raise ValueError("Currently only one global Emin_det implemented.")
            for s in seasons:
                try:
                    _Emin_det = Parameter.get_parameter(f"Emin_det_{s.P}").value.to(
                        u.GeV
                    )
                    Emin_det[s] = _Emin_det
                except ValueError:
                    raise ValueError("Emin_det not defined for all seasons.")

        try:
            roi = ROI.STACK[0]
        except IndexError:
            raise ValueError("No ROI on stack.")

        ra = np.hstack([events.ra[s.P] * u.rad for s in seasons])
        dec = np.hstack([events.dec[s.P] * u.rad for s in seasons])
        reco_energy = np.hstack([events.reco_energy[s.P] * u.GeV for s in seasons])
        types = np.hstack([events.ra[s.P].size * [s.S] for s in seasons])
        mjd = np.hstack([events.mjd[s.P] for s in seasons])

        # Conversion from 50% containment to 68% is already done in RealEvents
        ang_err = np.hstack([events.ang_err[s.P] * u.deg for s in seasons])
        coords = SkyCoord(ra=ra, dec=dec, frame="icrs")

        if isinstance(roi, CircularROI):
            mask = np.nonzero(
                (coords.separation(roi.center).deg * u.deg < roi.radius)
                & (mjd >= roi.MJD_min)
                & (mjd <= roi.MJD_max)
                & (reco_energy > Emin_det)
            )
        elif isinstance(roi, RectangularROI):
            if roi.RA_min > roi.RA_max:
                mask = np.nonzero(
                    (
                        (dec <= roi.DEC_max)
                        & (dec >= roi.DEC_min)
                        & ((ra >= roi.RA_min) | (ra <= roi.RA_max))
                        & (reco_energy >= Emin_det)
                        & (mjd >= roi.MJD_min)
                        & (mjd <= roi.MJD_max)
                    )
                )
            else:
                mask = np.nonzero(
                    (
                        (dec <= roi.DEC_max)
                        & (dec >= roi.DEC_min)
                        & (ra >= roi.RA_min)
                        & (ra <= roi.RA_max)
                        & (reco_energy >= Emin_det)
                        & (mjd >= roi.MJD_min)
                        & (mjd <= roi.MJD_max)
                    )
                )
        mjd = Time(mjd, format="mjd")

        return cls(
            reco_energy[mask],
            coords[mask],
            types[mask],
            ang_err[mask],
            mjd[mask],
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
