import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import ligo.skymap.plot

from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.utils.roi import (
    ROI,
    RectangularROI,
    CircularROI,
    FullSkyROI,
    ROIList,
)
from hierarchical_nu.source.source import Sources, PointSource
from hierarchical_nu.utils.plotting import SphericalCircle
from hierarchical_nu.detector.icecube import (
    Refrigerator,
    IC40,
    IC59,
    IC79,
    IC86_I,
    IC86_II,
)
from icecube_data_reader.event_types import EventType

from icecube_data_reader.events import IceTrackDR2Events

from time import time as thyme
import logging
from pathlib import Path
import os

from typing import List, Union
import numpy.typing as npt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""
class SingleEvent:

    @u.quantity_input
    def __init__(
        self,
        energy: u.GeV,
        coord: SkyCoord,
        type,
        ang_err: u.deg,
        mjd: Time,
    ):

        self._energy = energy
        self._mjd = np.atleast_1d(mjd)
        coord.representation_type = "spherical"
        self._coord = coord
        coord.representation_type = "cartesian"
        self._unit_vector = np.array([coord.x.value, coord.y.value, coord.z.value]).T
        self._coord.representation_type = "spherical"
        self._type = type
        self._ang_errs = ang_err

    @property
    def energy(self):
        return self._energy

    @property
    def ang_err(self):
        return self._ang_errs

    @property
    def coord(self):
        return self._coord

    @property
    def mjd(self):
        return self._mjd
"""


class Events(IceTrackDR2Events):
    """
    Events class for the storage of event observables
    """

    @u.quantity_input
    def __init__(
        self,
        energy: u.GeV,
        coord: SkyCoord,
        event_type: np.ndarray,
        ang_err: u.deg,
        mjd: Time,
    ):
        """
        Events class for the storage of event observables
        :param energies: Energies of events
        :param coords: Coords of events, instance of SkyCoord
        :param types: event types, e.g IC40
        :param ang_errs: angular uncertainties
        :param mjd: Arrival times of events in MJD
        """

        super().__init__(energy, coord, event_type, ang_err, mjd)

    """
    def __getitem__(self, i):
        event = SingleEvent(
            self.energies[i],
            self.coords[i],
            self.types[i],
            self.ang_errs[i],
            self.mjd[i],
        )
        return event
    """

    '''
    @classmethod
    def from_file(
        cls,
        filename,
        group_name=None,
        scramble_ra: bool = False,
        scramble_mjd: bool = False,  # TODO implement
        seed: int = 42,
        apply_spatial_cuts: bool = True,
        apply_temporal_cuts: bool = False,
        apply_Emin_det: bool = True,
    ):
        """
        Load events from simulated .h5 file.
        :param filename: Filename of event file
        :param group_name: Optional group name of event group in event file
        :param scramble_ra: Set to True if right ascension should be scrambled upon loading
        :param scramble_mjd: Not implemented yet, should scramble MJD
        :param seed: int, random seed for scrambling RA
        :param apply_spatial_cuts: Set to True if spatial ROI cuts should be applied
        :param apply_temporal_cuts: Set to True if temporal ROI cuts should be applied
        :param apply_Emin_det: Set to False if Emin_det should not be applied
        """

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

        mjd = Time(mjd, format="mjd")

        coords.representation_type = "spherical"

        dec = coords.dec.rad * u.rad
        ra = coords.ra.rad * u.rad
        if scramble_ra:
            logger.warning(
                "Scrambling RA, only sensible for simulations of the entire sky."
            )
            rng = np.random.default_rng(seed=seed)
            ra = rng.random(ra.size) * 2 * np.pi * u.rad
            coords = SkyCoord(ra=ra, dec=dec, frame="icrs")

        if not apply_spatial_cuts and not apply_temporal_cuts and not apply_Emin_det:
            events = cls(energies, coords, types, ang_errs, mjd)
            events._idxs = np.full(events.N, True)
            if events.N == 0:
                logger.warning("No events selected, check your simulation.")

            return events

        if ROIList.STACK:
            logger.info("Applying ROIs to event selection")

            mask = cls.apply_ROIS(
                coords,
                mjd,
                skip_time=not apply_temporal_cuts,
                skip_direction=not apply_spatial_cuts,
            )

            idxs = np.logical_or.reduce(mask)

            events = cls(
                energies[idxs], coords[idxs], types[idxs], ang_errs[idxs], mjd[idxs]
            )
            # Add the selection mask for easier comparison between simulations and fits
            # when using a subselection of the data
            events._idxs = idxs
        else:
            logger.info("Applying no ROIs to event selection")
            events = cls(energies, coords, types, ang_errs, mjd)
            events._idxs = np.full(events.N, True)

        # Apply energy cuts
        if apply_Emin_det:
            try:
                _Emin_det = Parameter.get_parameter("Emin_det")
                mask = events.energies >= _Emin_det.value
                # logger.info(f"Applying Emin_det={_Emin_det.value} to event selection.")
                logger.info(f"Applying Emin_det={_Emin_det.value} to event selection.")

            except ValueError:
                _types = np.unique(events.types)
                mask = np.full(events.energies.size, True)
                for _t in _types:
                    try:
                        _Emin_det = Parameter.get_parameter(
                            f"Emin_det_{Refrigerator.stan2python(_t)}"
                        )
                        mask[events.types == _t] = (
                            events.energies[events.types == _t] >= _Emin_det.value
                        )
                        logger.info(
                            f"Applying Emin_det={_Emin_det.value} to event selection."
                        )
                    except ValueError:
                        pass

            events.select(mask)
        if events.N == 0:
            logger.warning("No events selected, check your ROI and MJD")

        return events
    '''

    def export_to_csv(self, basepath):
        """
        Create new csv files with similar formatting to 10 year point source data
        :param basepath: Directory in which to save the .csv files
        """
        raise NotImplementedError("TODO: needs to be implemented")
        header = "log10(E/GeV)\tAngErr[deg]\tRA[deg]\tDec[deg]"
        energy = np.log10(self.energies.to_value(u.GeV))
        ang_errs = self.ang_errs.to_value(u.deg)
        self.coords.representation_type = "spherical"
        ra = self.coords.ra.deg
        dec = self.coords.dec.deg
        fmt = ("%3.2f", "%3.2f", "%3.3f", "%3.3f")
        for t in np.unique(self.types):
            filename = basepath / Path(f"{Refrigerator.stan2python(t)}_exp.csv")
            mask = self.types == t
            array = np.vstack((energy, ang_errs, ra, dec)).T[mask]
            np.savetxt(filename, array, fmt=fmt, delimiter="\t\t", header=header)
        return

    def get_tags(self, sources: Sources):
        """
        Idea: each event gets one PS (smallest distance), assumes that CircularROIs do not overlap
        :param sources: instance of `Sources`
        """

        logger.warning("Applying tags is experimental.")
        self.coords.representation_type = "spherical"
        ps_ra = np.array([_.ra.to_value(u.deg) for _ in sources.point_source]) * u.deg
        ps_dec = np.array([_.dec.to_value(u.deg) for _ in sources.point_source]) * u.deg
        ps_coords = SkyCoord(ra=ps_ra, dec=ps_dec, frame="icrs")
        tags = []
        for coord in self.coords:
            ang_dist = coord.separation(ps_coords).deg
            # Take source with smallest angular separation
            tags.append(np.argmin(ang_dist))
        return tags

    '''
    def to_icecube_tools(self):
        """
        Return `icecube_tools.utils.data.SimEvents` object from current events.
        """

        from icecube_tools.utils.data import SimEvents, dddict

        it_types = np.unique([IC40.P, IC59.P, IC79.P, IC86_I.P, IC86_II.P])
        hnu_types = np.unique([Refrigerator.stan2python(_) for _ in self._types])
        # Check that only r2021 events are present
        if not np.isin(hnu_types, it_types):
            raise ValueError("Erroneous event types used.")

        reco_energies = dddict()
        ang_errs = dddict()
        ras = dddict()
        decs = dddict()

        fields = [reco_energies, ang_errs, ras, decs]
        # create empty lists
        for et in hnu_types:
            for field in fields:
                field[et] = []
        coords = self.coords
        coords.representation_type = "spherical"
        for E, ang_err, coord, t in zip(
            self.energies, self.ang_errs, coords, self.types
        ):
            p = Refrigerator.stan2python(t)
            reco_energies[p].append(E.to_value(u.GeV))
            ang_errs[p].append(ang_err.to_value(u.deg))
            ras[p].append(coord.ra.rad)
            decs[p].append(coord.dec.rad)

        for et in hnu_types:
            ras[et] = np.array(ras[et])
            decs[et] = np.array(decs[et])
            ang_errs[et] = np.array(ang_errs[et])
            reco_energies[et] = np.array(reco_energies[et])

        events = SimEvents()
        events._ra = ras
        events._dec = decs
        events._reco_energy = reco_energies
        events._ang_err = ang_errs

        return events
    '''

    @classmethod
    def from_event_files(
        cls,
        *seasons: EventType,
        scramble_ra: bool = False,
        scramble_mjd: bool = False,
        seed: int = 42,
        apply_spatial_cuts: bool = True,
        apply_temporal_cuts: bool = True,
        apply_Emin_det: bool = True,
    ):
        """
        Load events from the 2021 data release
        :param seasons: arbitrary number of `EventType` identifying detector seasons of r2021 release.
        :param scramble_ra: Set to true if RA should be randomised
        :param seed: int, random seed for RA scrambling
        :param apply_spatial_cuts: if True, apply spatial cuts
        :param apply_temporal_cuts: if True, apply_temporal cuts
        :param apply_Emin_det if True, apply Emin_det cuts
        :return: :class:`hierarchical_nu.events.Events`
        """

        if not len(seasons) == len(set(seasons)):
            raise ValueError("Detector season is provided twice.")

        # Borrow from icecube_tools
        # Already exclude low energy events here, would be quite difficult later on
        events = super().from_event_files(*seasons)
        try:
            _Emin_det = Parameter.get_parameter("Emin_det").value
            if apply_Emin_det:
                events.apply_energy_cut(Emin=_Emin_det)
        except ValueError:
            # Create a dict of masks for each season
            # if apply_Emin_det:
            if False:
                # TODO Implement...

                for s in seasons:
                    try:
                        _Emin_det = Parameter.get_parameter(
                            f"Emin_det_{s.P}"
                        ).value.to_value(u.GeV)
                        mask[s] = (
                            events.reco_energy[events.event_type == s] >= _Emin_det
                        )
                    except ValueError:
                        raise ValueError("Emin_det not defined for all seasons.")

        # if scramble_mjd:
        if False:
            # TODO implement
            events.scramble_mjd()
            lt = Uptime()
            ic86 = ["IC86_II", "IC86_III", "IC86_IV", "IC86_V", "IC86_VI", "IC86_VII"]

            # I have no one but myself to blame for this
            lt._data["IC86_II"] = np.vstack([lt._data[s] for s in ic86])

            for s in seasons:
                intervals = np.sum(lt._data[s.P], axis=1)

                # Sample new good time intervals for each event,
                # weight is the intervals length
                idxs = rng.choice(
                    np.arange(intervals.size),
                    size=np.sum(types == s.S, dtype=int),
                    p=intervals / np.sum(intervals),
                )
                start = lt._data[s.P][idxs, 0]
                end = lt._data[s.P][idxs, 1]
                # scramble arrival time within each season separately
                mjd[types == s.S] = rng.uniform(low=start, high=end, size=idxs.size)

        if scramble_ra:
            events.scramble_ra(seed=seed)

        if apply_spatial_cuts or apply_temporal_cuts:
            mask = cls.apply_ROIS(
                events,
                skip_time=not apply_temporal_cuts,
                skip_direction=not apply_spatial_cuts,
            )

            idxs = np.logical_or.reduce(mask)
            events.select(idxs)

        return events

    def merge(self, events):
        """
        Merge events with a different instance of `Events`.
        Returns newly created instance.
        :param events: Instance of Events to merge with
        """

        if not isinstance(events, Events):
            raise TypeError("`events` must be instance of `Events`")

        self.coords.representation_type = "spherical"
        events.coords.representation_type = "spherical"
        ra = np.hstack((self.coords.ra.deg * u.deg, events.coords.ra.deg * u.deg))
        dec = np.hstack((self.coords.dec.deg * u.deg, events.coords.dec.deg * u.deg))
        coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
        energies = np.hstack([self.energies, events.energies])
        ang_errs = np.hstack([self.ang_errs, events.ang_errs])
        types = np.hstack([self.types, events.types])
        mjd = Time(np.hstack([self.mjd.mjd, events.mjd.mjd]), format="mjd")

        return Events(energies, coords, types, ang_errs, mjd)

    @u.quantity_input
    def plot_radial_excess(
        self, position: Union[SkyCoord, PointSource], radius: u.deg = 5 * u.deg
    ):
        """
        Plot histogram of radial distance to a source located at center.
        Bin edges are equdistant in angle squared such that equal areas in polar coordinates
        (assuming Euclidian space for small angles) are covered by each bin.
        :param position: SkyCoord of center or PointSource instance
        :param radius: Max radius of histogram
        """

        if isinstance(position, PointSource):
            center_coords = SkyCoord(ra=position.ra, dec=position.dec, frame="icrs")
        elif isinstance(position, SkyCoord):
            center_coords = position
        else:
            raise ValueError

        r2_bins = np.arange(
            0.0, np.power(radius.to_value(u.deg), 2) + 1.0 / 3.0, 1.0 / 3.0
        )
        sep = center_coords.separation(self.coords).deg

        fig, ax = plt.subplots()
        n, bins, _ = ax.hist(sep**2, r2_bins, histtype="step")
        ax.set_xlabel("$\Psi^2$ [deg$^2$]")
        ax.set_ylabel("counts")

        return (
            fig,
            ax,
            n,
            bins,
        )

    @classmethod
    def apply_ROIS(
        cls,
        coords: SkyCoord,
        mjd: Time,
        skip_time: bool = False,
        skip_direction: bool = False,
    ):
        """
        Returns list of mask, one mask for each ROI on stack
        """

        ra = coords.icrs.ra
        dec = coords.icrs.dec

        mask = []
        for roi in ROIList.STACK:
            time = (mjd.mjd <= roi.MJD_max) & (mjd.mjd >= roi.MJD_min)
            if isinstance(roi, CircularROI):
                direction = roi.radius >= roi.center.separation(coords)
            else:
                if roi.RA_min > roi.RA_max:
                    direction = (
                        (dec <= roi.DEC_max)
                        & (dec >= roi.DEC_min)
                        & ((ra >= roi.RA_min) | (ra <= roi.RA_max))
                    )

                else:
                    direction = (
                        (dec <= roi.DEC_max)
                        & (dec >= roi.DEC_min)
                        & (ra >= roi.RA_min)
                        & (ra <= roi.RA_max)
                    )
            if skip_time and skip_direction:
                mask.append([True] * coords.size)
            elif skip_time:
                mask.append(direction)
            elif skip_direction:
                mask.append(time)
            else:
                mask.append(time & direction)

        return mask
