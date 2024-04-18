# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: hi_nu
#     language: python
#     name: python3
# ---

from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors
from hierarchical_nu.source.source import Sources, PointSource, DetectorFrame
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior, FluxPrior, LuminosityPrior, IndexPrior
from hierarchical_nu.utils.plotting import SphericalCircle
from hierarchical_nu.utils.roi import CircularROI, ROIList
from hierarchical_nu.detector.icecube import IC86_II, IC40, IC59, IC79, IC86_I
from hierarchical_nu.detector.r2021 import R2021LogNormEnergyResolution, R2021EnergyResolution
from hierarchical_nu.backend import DistributionMode
from hierarchical_nu.detector.input import mceq
from icecube_tools.utils.data import Uptime
import numpy as np
import matplotlib.pyplot as plt
import arviz as av
import h5py
import ligo.skymap.plot
from hierarchical_nu.source.source import uv_to_icrs
from hierarchical_nu.utils.config_parser import ConfigParser
from hierarchical_nu.utils.config import HierarchicalNuConfig

config = HierarchicalNuConfig.from_path("hnu_config.yml")

parser = ConfigParser(config)

sources = parser.sources

parser.ROI
ROIList.STACK

events = parser.events
events.N

fig, axs = events.plot_energy(sources[0])
fig.savefig("event_display.pdf", dpi=150)

fig, axs = events.plot_radial_excess(sources[0])
fig.savefig("event_radial_excess.pdf", dpi=150)

# +
Parameter.clear_registry()
src_index = Parameter(
    value=2.2,
    name="src_index",
    fixed=False,
    par_range=(1., 4.),
)
diff_index = Parameter(
    value=2.,
    name="diff_index",
    fixed=False,
    par_range=(1., 4.),
)
L = Parameter(
    value=1e47 * (u.erg / u.s),
    name="luminosity",
    fixed=True,
    par_range=(0, 1e55) * (u.erg / u.s),
)
diffuse_norm = Parameter(
    value=2.26e-13 / u.GeV / u.s / u.m**2, name="diffuse_norm", fixed=True, par_range=(0, np.inf)
)

# True energy range and normalisation
Enorm = Parameter(1e5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1e2 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1e8 * u.GeV, "Emax", fixed=True)
Emin_diff = Parameter(
    1e2 * u.GeV, "Emin_diff", fixed=True
)
Emax_diff = Parameter(
    1e8 * u.GeV, "Emax_diff", fixed=True
)
Emin_src = Parameter(
    1e2 * u.GeV, "Emin_src", fixed=True
)
Emax_src = Parameter(
    1e8 * u.GeV, "Emax_src", fixed=True
)

Emin_det = Parameter(
    3e2 * u.GeV, "Emin_det", fixed=True
)

# -

dm = parser.detector_model
dm

obs_time = parser.obs_time
obs_time

sim = parser.create_simulation(sources, dm, obs_time)

sim.precomputation()
print(sim._get_expected_Nnu(sim._get_sim_inputs()))
print(sim._expected_Nnu_per_comp)


sim.generate_stan_code()
sim.compile_stan_code()

sim.run()

sim.show_skymap()
sim.show_spectrum()

fit = parser.create_fit(sources, events, dm, obs_time)

fit.precomputation(sim._exposure_integral)
fit.generate_stan_code()
fit.compile_stan_code()

fit.run(
    show_progress=True,
    inits={"L": 1e49, "src_index": 2.2, "F_diff": 1e-4, "F_atmo": 0.3, "diff_index": 2.2, "E": [1e5] * fit.events.N},
    chains=1,
    save_profile=True,
    show_console=True,
    seed=43,
)

print(fit._fit_output.diagnose())

fig, axs = fit.plot_energy_and_roi()
fig.savefig("energy_and_roi.pdf", dpi=150)

fig, axs = fit.plot_trace_and_priors(fit._def_var_names+["Nex_src", "Nex_atmo", "Nex_diff"])
fig.savefig("trace.pdf", dpi=150, bbox_inches="tight")

# +
keys = ["L", "src_index", "Nex_src"]
label = [r"$L~[erg/s]$", r"$\gamma$", "Nex"]
transformations = [lambda x: x * (u.GeV / u.erg).to(1), lambda x: x, lambda x: x]
CL = [0.5, 0.683]

hdi = {key: [av.hdi(transformations[c](fit._fit_output.stan_variable(key).T), hdi_prob=_).flatten() for _ in CL] for c, key in enumerate(keys)}
kdes = {key: av.kde(transformations[c](fit._fit_output.stan_variable(key))) for c, key in enumerate(keys)}
# -

ref_nex = np.array([10.00, 4.2, 5.2])
ref_index = np.array([2.2, 0.3, 0.3])   # https://arxiv.org/pdf/2109.05818.pdf as reference, but they use a sliding gaussian to actually determine the time window, so be careful
ref = {"src_index": ref_index, "Nex_src": ref_nex}

# +
fig, axs = plt.subplots(nrows=len(keys), ncols=1, figsize=(2.80278, 4), gridspec_kw={"hspace": .80})

for c, key in enumerate(keys):
    ax = axs[c]
    kde = kdes[key]
    HDI = hdi[key]
    ax.plot(*kde)
    for HDI in hdi[key]:
        start = np.digitize(HDI[0], kde[0]) - 1
        stop = np.digitize(HDI[1], kde[0])
        ax.fill_between(kde[0][start:stop], np.zeros_like(kde[0][start:stop]), kde[1][start:stop], color="C0", alpha=0.3)
    try:
        errs = ref[key]
        ax.errorbar(errs[0], np.sum(ax.get_ylim())*0.45, xerr=errs[1:3][:, np.newaxis], color="black", capsize=4, fmt="x")
    except KeyError:
        pass
    ax.set_yticks([])
    ax.set_xlabel(label[c])
    
# -

fig.savefig("compare_to_icecube.pdf", dpi=150)


