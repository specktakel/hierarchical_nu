import numpy as np

import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import ligo.skymap.plot

from hierarchical_nu.utils.roi import (
    CircularROI,
    ROIList,
)
from hierarchical_nu.source.source import Sources
from hierarchical_nu.utils.plotting import SphericalCircle

from icecube_data_reader.events import IceTrackDR2Events

import logging
from pathlib import Path


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
        self, position: SkyCoord, radius: u.deg = 5 * u.deg
    ):
        """
        Plot histogram of radial distance to a source located at center.
        Bin edges are equdistant in angle squared such that equal areas in polar coordinates
        (assuming Euclidian space for small angles) are covered by each bin.
        :param position: SkyCoord of center or PointSource instance
        :param radius: Max radius of histogram
        """

        center_coords = position

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

    
    def apply_ROIS(
        self,
        skip_time: bool = False,
        skip_direction: bool = False,
    ):
        """
        Returns list of mask, one mask for each ROI on stack
        """

        ra = self.coords.icrs.ra
        dec = self.coords.icrs.dec

        mask = []
        for roi in ROIList.STACK:
            time = (self.mjd.mjd <= roi.MJD_max) & (self.mjd.mjd >= roi.MJD_min)
            if isinstance(roi, CircularROI):
                direction = roi.radius >= roi.center.separation(self.coords)
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
                mask.append([True] * self.N)
            elif skip_time:
                mask.append(direction)
            elif skip_direction:
                mask.append(time)
            else:
                mask.append(time & direction)

        mask = np.logical_or.reduce(mask)

        self.select(mask)
