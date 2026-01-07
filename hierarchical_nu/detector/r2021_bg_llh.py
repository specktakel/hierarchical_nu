from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union

from icecube_tools.utils.data import data_directory, available_datasets


base_path = Path(data_directory) / Path(available_datasets["20210126"]["dir"])


class R2021BackgroundLLH:

    def __init__(self, season: str = "IC86_II"):
        from skyllh.i3.backgroundpdf import DataBackgroundI3SpatialPDF

        from skyllh.core.config import Config
        from skyllh.i3.config import add_icecube_specific_analysis_required_data_fields

        from skyllh.datasets.i3.PublicData_10y_ps import create_dataset_collection
        from skyllh.core.timing import TimeLord
        from skyllh.analyses.i3.publicdata_ps.backgroundpdf import (
            PDDataBackgroundI3EnergyPDF,
        )
        from skyllh.core.smoothing import (
            BlockSmoothingFilter,
        )

        cfg = Config()
        add_icecube_specific_analysis_required_data_fields(cfg)

        cfg["datafields"].pop("run", None)
        cfg["repository"]["download_from_origin"] = False
        dsc = create_dataset_collection(
            cfg=cfg,
            base_path=base_path,
        )

        if season == "IC86_II":
            season = "IC86_II-VII"

        ds = dsc[season]
        data = ds.load_and_prepare_data(
            keep_fields=None,
            dtc_dict=None,
            dtc_except_fields=None,
        )
        spatial_pdf = DataBackgroundI3SpatialPDF(
            cfg=cfg,
            data_exp=data.exp,
            sin_dec_binning=ds.get_binning_definition("sin_dec"),
        )
        energy_pdf = PDDataBackgroundI3EnergyPDF(
            cfg=cfg,
            data_exp=data.exp,
            logE_binning=ds.get_binning_definition("log_energy"),
            sinDec_binning=ds.get_binning_definition("sin_dec"),
            smoothing_filter=BlockSmoothingFilter(nbins=1),
            kde_smoothing=False,
        )
        dec_binning = ds.get_binning_definition("sin_dec")
        logEreco_binning = ds.get_binning_definition("log_energy")
        self._energy_pdf = energy_pdf
        self._spatial_pdf = spatial_pdf
        self._dec_binning = dec_binning
        self._logEreco_binning = logEreco_binning

    def energy_integral(self, sin_dec: float, log10_elow: float, log10_ehigh: float):
        elow = log10_elow
        ehigh = log10_ehigh

        pdf = self._energy_pdf
        edges = pdf.binnings[0].binedges
        elow = max(elow, edges.min())
        ehigh = min(ehigh, edges.max())
        sin_dec_idx = np.digitize(sin_dec, pdf.binnings[1].binedges) - 1
        if sin_dec_idx >= pdf.binnings[1].binedges.max():
            sin_dec_idx = pdf.binnings[1].bincenters.size - 1
        new_edges = np.unique(
            np.hstack((np.atleast_1d(elow), edges, np.atleast_1d(ehigh)))
        )

        new_edges = np.unique(
            np.hstack((np.atleast_1d(elow), edges, np.atleast_1d(ehigh)))
        )

        idx_l = np.digitize(elow, edges) - 1
        idx_h = np.digitize(ehigh, edges, right=True)
        diff = np.diff(
            new_edges[(new_edges <= ehigh) & (new_edges >= elow)]
        )  # this is the corrsponding deltaE
        integral = np.sum(diff * pdf._hist_logE_sinDec[idx_l:idx_h, sin_dec_idx])
        return integral

    def prob_omega(self, sin_dec: Union[float, np.ndarray]):
        return 0.5 / np.pi * np.exp(self._spatial_pdf._log_spline(sin_dec))

    def prob_ereco_given_sindec(
        self, ereco: Union[float, np.ndarray], sin_dec: Union[float, np.ndarray]
    ):
        pdf = self._energy_pdf
        edges = pdf.binnings[0].binedges
        ereco_idx = np.digitize(ereco, edges) - 1
        """
        try:
            idxs = ereco >= edges.max()
            ereco_idx[idxs] = edges.size - 2
            ereco_idx[ereco <= edges.min()] = 0
        except:
            if ereco >= edges.max():
                ereco_idx = edges.size - 2
            if ereco <= edges.min():
                ereco_idx = 0
        """
        edges = pdf.binnings[1].binedges
        sin_dec_idx = np.digitize(sin_dec, edges) - 1
        """
        try:
            idxs = sin_dec_idx >= pdf.binnings[1].binedges.max()
            sin_dec_idx[idxs] = pdf.binnings[1].bincenters.size - 1
        except (IndexError, TypeError):
            if sin_dec_idx >= pdf.binnings[1].binedges.max():
                sin_dec_idx = pdf.binnings[1].bincenters.size - 1
        """
        p_ereco_given_sindec = pdf._hist_logE_sinDec[(ereco_idx, sin_dec_idx)]
        return p_ereco_given_sindec

    def prob_ereco_and_omega(
        self, ereco: Union[float, np.ndarray], sin_dec: Union[float, np.ndarray]
    ):
        p_sin_dec = self.prob_omega(sin_dec)
        p_ereco_given_sindec = self.prob_ereco_given_sindec(ereco, sin_dec)
        return p_ereco_given_sindec * p_sin_dec
