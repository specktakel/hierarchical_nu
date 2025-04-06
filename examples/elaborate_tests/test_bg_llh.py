import pytest
import numpy as np
from pathlib import Path

from hierarchical_nu.detector.r2021_bg_llh import R2021BackgroundLLH
from skyllh.core.config import Config
from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection
from icecube_tools.utils.data import data_directory, available_datasets
from skyllh.analyses.i3.publicdata_ps.time_integrated_ps import create_analysis
from skyllh.core.source_model import PointLikeSource


@pytest.fixture
def ana():
    base_path = Path(data_directory) / Path(available_datasets["20210126"]["dir"])

    cfg = Config()
    dsc = create_dataset_collection(
        cfg=cfg, base_path=base_path
    )

    datasets = [dsc["IC86_II-VII"]]
    source = PointLikeSource(ra=np.deg2rad(77.35), dec=np.deg2rad(5.7))
    ana = create_analysis(cfg=cfg, datasets=datasets, source=source)
    events_list = [data.exp for data in ana.data_list]
    ana.initialize_trial(events_list)
    return ana

@pytest.fixture
def spatial_skyllh(ana):
    # Set up all skyllh related bits, TODO make fixture
    
    spatial_bg = ana._pdfratio_list[0].pdfratio1.bkg_pdf

    # this is the values of the selected events
    tdm = ana._tdm_list[0]
    pdf = spatial_bg.get_pd(tdm)[0]
    sin_dec = tdm.get_data('sin_dec')

    return pdf, sin_dec


@pytest.fixture
def llh():
    llh = R2021BackgroundLLH()
    return llh
def test_spatial_hnu(spatial_skyllh, llh):
    pdf, sin_dec = spatial_skyllh

    hnu_llh = llh.prob_omega(sin_dec)
    # assert pdf.size == sin_dec.size
    assert pdf == pytest.approx(hnu_llh, rel=1e-6)


@pytest.fixture
def energy_skyllh(ana):
    energy_pdf = ana._pdfratio_list[0].pdfratio2.bkg_pdf

    tdm = ana._tdm_list[0]
    pdf = energy_pdf.get_pd(tdm)[0]
    ereco = tdm.get_data("log_energy")
    sin_dec = tdm.get_data("sin_dec")

    return pdf, ereco, sin_dec

def test_ereco_hnu(energy_skyllh, llh):
    pdf, ereco, sin_dec = energy_skyllh

    hnu_llh = llh.prob_ereco_given_sindec(ereco, sin_dec)

    assert pdf == pytest.approx(hnu_llh, rel=1e-6)

