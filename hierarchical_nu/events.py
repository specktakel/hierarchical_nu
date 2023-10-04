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

import logging

from typing import List

logger = logging.getLogger(__name__)

TRACKS = 0
CASCADES = 1

IC40 = 0
IC59 = 1
IC79 = 2
IC86_I = 3
IC86_II = 4

periods = {
    "IC40": IC40,
    "IC59": IC59,
    "IC79": IC79,
    "IC86_I": IC86_I,
    "IC86_II": IC86_II,
}

# Translate IRF period into integer from 0 to 4 (IC40 to IC86_II)


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
        periods: List[str] = None,
    ):
        """
        Events class for the storage of event observables
        """

        self._recognised_types = [TRACKS, CASCADES]

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

        if periods is not None:
            self._periods = periods

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
        p: str,
        Emin_det: u.GeV = 1 * u.GeV,
        **kwargs,
    ):
        """
        Load events from the 2021 data release
        :param p: string of period to be loaded.
        :param kwargs: kwargs passed to make an event selection, see `icecube_tools` documentation for details
        :return: :class:`hierarchical_nu.events.Events`
        """

        from icecube_tools.utils.data import RealEvents

        # Borrow from icecube_tools
        use_all = kwargs.pop("use_all", True)
        events = RealEvents.from_event_files(p, use_all=use_all)

        # Check if minimum detected energy is currently loaded as parameter
        try:
            Emin_det = Parameter.get_parameter("Emin_det_t").value.to(u.GeV)
            logger.warning(f"Overwriting Emin_det with {Emin_det}")
        except ValueError:
            try:
                Emin_det = Parameter.get_parameter("Emin_det").value.to(u.GeV)
                logger.warning(f"Overwriting Emin_det with {Emin_det}")
            except ValueError:
                pass

        try:
            roi = ROI.STACK[0]
        except IndexError:
            raise ValueError("No ROI on stack.")

        # events.restrict(**kwargs) # Do this completely in hnu, icecube_tools lacks the RA-wrapping right now, TODO...
        # Read in relevant data
        ra = events.ra[p] * u.rad
        dec = events.dec[p] * u.rad
        reco_energy = events.reco_energy[p] * u.GeV
        period = ra.size * [p]
        mjd = events.mjd[p]

        # Conversion from 50% containment to 68% is already done in RealEvents
        ang_err = events.ang_err[p] * u.deg
        types = np.array(ra.size * [TRACKS])
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
            p,
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
