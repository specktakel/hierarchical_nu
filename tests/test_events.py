from hierarchical_nu.events import Events
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource, DetectorFrame
from hierarchical_nu.utils.roi import CircularROI, RectangularROI, ROIList, FullSkyROI
from hierarchical_nu.detector.icecube import Refrigerator, IC86, IC40
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np

import pytest

events_file_name = "test_event_read_write.h5"


def test_event_class(output_directory):
    FullSkyROI()
    Parameter(1e1 * u.GeV, "Emin_det", fixed=True)

    N_in = 50
    energies_in = np.linspace(1, 10, N_in) * u.TeV

    coords_in = SkyCoord(
        ra=np.linspace(0, 2 * np.pi, N_in) * u.rad,
        dec=np.linspace(-np.pi / 2, np.pi / 2, N_in) * u.rad,
    )

    types_in = np.full(N_in, IC86)
    ang_errs_in = np.ones(N_in) * u.deg
    mjd_in = Time(np.tile(99, N_in), format="mjd")

    events_in = Events(
        energies=energies_in,
        coords=coords_in,
        types=types_in,
        ang_errs=ang_errs_in,
        mjd=mjd_in,
    )

    events_file = output_directory / events_file_name
    events_in.to_file(events_file, overwrite=True)
    events_out = Events.from_file(events_file)

    assert np.all(events_out.energies == energies_in)

    assert np.all(events_out.types == types_in)
    assert np.all(events_out.ang_errs == ang_errs_in)
    assert np.all(events_out.mjd == mjd_in)

    events_out = Events.from_file(events_file)
    N = events_out.N

    events_out.remove(1)

    assert events_out.N == N - 1

    mask = events_out.energies > 5 * u.TeV
    events_out.select(mask)

    assert np.all(events_out.energies > 5 * u.TeV)
    assert events_out.N < N

def test_event_cuts():
    coord = SkyCoord(ra=77.6 * u.deg, dec=5.7*u.deg)
    Emin_det = Parameter(5e4 * u.GeV, "Emin_det", fixed=True)
    roi = CircularROI(coord, radius=5 * u.deg)

    events = Events.from_event_files()
    events.apply_ROIS()

    assert np.all(coord.separation(events.coords)<=roi.radius)

    events.apply_Emin_det()

    assert np.all(events.energies >= Emin_det.value)

def test_event_tag():
    c1 = SkyCoord(ra=77.6 * u.deg, dec=5.7*u.deg)
    c2 = SkyCoord(ra=40.6696 * u.deg, dec = -0.01329*u.deg)
    src_index = Parameter(2.2, "src_index")
    L = Parameter(1e42*u.erg / u.s, "L")
    Emin_src = Parameter(1e2 * u.GeV, "Emin_src")
    Emax_src = Parameter(1e8 * u.GeV, "Emax_src")
    z1 = 0.03365
    z2 = 0.00458

    events = Events.from_event_files(IC40)


    sources = Sources()

    for c, (coord, z) in enumerate(zip([c1, c2], [z1, z2])):
        dec = coord.dec
        ra = coord.ra
        sources.add(
            PointSource.make_powerlaw_source(
            f"ps_{c}", dec, ra, L, src_index, z, Emin_src, Emax_src, DetectorFrame,
            )
        )

    tags = events.get_tags(sources)
    seps1 = c1.separation(events.coords).deg
    seps2 = c2.separation(events.coords).deg
    assert np.all(seps1[tags==0] <= seps2[tags==0])
    assert np.all(seps1[tags==1] >= seps2[tags==1])
    
    
