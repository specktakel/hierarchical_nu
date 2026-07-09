---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: ic-data
    language: python
    name: python3
---

## Analysing experimental data


The 10 year PS data set provides multiple IRFs and events recorded in the particular detector configurations, starting with `IC40` and ending with `IC86_VII`, although the detector configuration did not change after `IC86_II`.

As an example we consider TXS0506+056 during the 2014/2015 flare.

```python
from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.utils.roi import CircularROI
# from hierarchical_nu.detector.input import mceq
import numpy as np
import ligo.skymap.plot
import h5py
from hierarchical_nu.detector.icecube import IC86
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.priors import Priors, LuminosityPrior, FluxPrior, NormalPrior, LogNormalPrior, LogUniformPrior, IndexPrior, NexPrior, Ignorance
from hierarchical_nu.fit import StanFit
from hierarchical_nu.source.source import PointSource, Sources, DetectorFrame
from arviz_stats import hdi
import matplotlib.pyplot  as plt
# from icecube_data_reader.events import IceTrackDR2Events as Events 
from hierarchical_nu.events import Events
import arviz_base as avb
import seaborn.objects as so
import healpy as hp
```

```python
events = Events.from_event_files(IC86)
```

First, we define the source and fit parameters, as already seen in `simulate_and_fit`. The value of e.g. `src_index` has no meaning in this context. Relevant for the fit is only the existance of the parameter itself. We can, however, use the parameter values to make decisions on which priors to choose by gauging how many events are to be expected by a certain choice of parameters.

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1, 4))
diff_index = Parameter(2.52, "diff_index", fixed=False, par_range=(1, 4))
L = Parameter(1e47 * (u.erg / u.s), "luminosity", fixed=True, 
              par_range=(0, 1E60) * (u.erg/u.s))
diffuse_norm = Parameter(2.26e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(1e-14, 1e-11)*(1/u.GeV/u.s/u.m**2))
z = 0.3365
Nex_src = Parameter(0, "Nex_src", fixed=True, par_range=(0., 100.))
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
Emin_src = Parameter(Emin.value, "Emin_src", fixed=True)
Emax_src = Parameter(Emax.value, "Emax_src", fixed=True)
Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)
```

```python
Emin_det = Parameter(3e2 * u.GeV, "Emin_det", fixed=True)
```

```python
# Single PS for testing and usual components
ra = np.deg2rad(77.35) * u.rad
dec = np.deg2rad(5.7) * u.rad
width = np.deg2rad(6) * u.rad
txs = SkyCoord(ra=ra, dec=dec, frame="icrs")

point_source = PointSource.make_powerlaw_source(
    "test", dec, ra, L, src_index, z, Emin_src, Emax_src, DetectorFrame,
)

my_sources = Sources()
my_sources.add(point_source)
my_sources.add_background(IC86)
#my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff) 
#my_sources.add_atmospheric_component(cache_dir=mceq)

```

```python
my_sources.sources
```

We now need to decide on the time period of observation (start and end times in MJD). The detector lifetime is automatically calculated from the "good time intervals" provided in the data release. Event selection respects the start and end times with which an ROI is instanciated.

From the [paper](https://arxiv.org/pdf/2101.09836.pdf) accompanying the data release, Table IV, we can guesstimate (Tmin, Tmax) = (56917, 57113). This is not the result of a flare analysis, but rather to include all contributing events.

Together with the source location we create an ROI.

```python
MJD_min=56917
MJD_max=57113

roi = CircularROI(txs, 5 * u.deg, MJD_min=MJD_min, MJD_max=MJD_max, apply_roi=True)
```

```python
events.apply_ROIS()
```

```python
events.N
```

A `LifeTime` instance computes the detector lifetime in the chosen time range.

```python
lt = LifeTime()

```

```python
lt = LifeTime()
lifetime = lt.lifetime_from_mjd(MJD_min, MJD_max, squeeze=True)
print(lifetime)
```

Returned is dictionary with event type as key and lifetime in years as value. We can re-use this return value to get a list of event types as input for the fit.


We take a small detour to setup a simulation. Through `sim._get_expected_Nnu` we are able to calculate the expected number of events.
Due to a mismatch between data and MC it is currently sensible to set a prior on the atmospheric flux that will account for the selected number of events, rather than the MCEq-simulated flux value.

Assuming all data is background, we rescale the atmospheric flux to produce the number of events in the data sample. Data taking is a Poisson counts experiment, meaning that $\sqrt{N}$ is the standard deviation of $N$.

```python
priors = Priors()
priors.src_index = IndexPrior(mu=2.5, sigma=1)
priors.Nex_src = NexPrior(NormalPrior, mu=6, sigma=10)
```

```python
priors.Nex_src.mu
```

```python
# Copy flux
'''
atmo_flux_int = my_sources.atmospheric.flux_model.total_flux_int

# N +/- N_std -> N_std is sigma on scale of events
# atmo_flux_int +/- sigma should yield N +/- N_std events
mu = 0.32 / u.m**2 / u.s
sigma = 0.1 / u.m**2 / u.s
priors.atmospheric_flux = FluxPrior(mu=mu, sigma=sigma)
'''
```

```python
event_types = list(lifetime.keys())
```

```python
event_types[0].model
```

```python
roi.DEC_max.to(u.deg), roi.DEC_min.to(u.deg)
```

```python
fit = StanFit(my_sources, event_types, events, lifetime, priors=priors, nshards=8)
```

```python
fit.precomputation(show_progress=True)
```

```python
fit.generate_stan_code()
fit.compile_stan_code()
```

Run the fit with some appropriate initial values. With 40 threads, this will take roughly 6 minutes (subject to change, hopefully for the better).

```python
fit.run(
    show_progress=True, inits={
        "L": 1e48,
        "src_index": 2.2,
        "diff_index": 2.2,
        "F_atmo": 0.3,
        "diffuse_norm": 2.2e-13,
        "E": [1e4] * fit.events.N,
        "Nex_src": [6]},
)
```

Confirm that there are no problems encountered by the HMC.

```python
fit.diagnose()
```

```python
np.average(fit["Nex_src"])
```

```python
fit.plot_energy_and_roi()
```

```python
# use assoc_prob to source: build new distribution of Enu by taking samples of all neutrino energies
# weighted by their assoc prob to the point source at that particular energy
```

```python
assoc_dist = fit._get_event_association_dist()
```

```python
assoc_dist.shape   # dims: #events, #source component, #sample
```

```python
###
# first sample N times an event from excess weighed by p_assoc[excess_mask]
# within each sampled event, sample one Enu based on assoc_dist within that event
```

```python
N_excess = np.ceil(hdi(fit["Nex_src"].flatten(), 0.68)[1]).astype(int)
N_excess = 12
excess_ps = p_assoc[mask_excess]
```

```python
p_assoc = np.array(fit._get_event_classifications())[:, 0]
mask_excess = np.argsort(p_assoc)[::-1][:N_excess]
mask_excess
```

```python
np.swapaxes(fit["E"][0][:, mask_excess], 1, 0)
```

```python

E_reshaped = fit["E"].reshape(fit.chains * fit.iterations, fit.events.N)
E_reshaped = np.swapaxes(E_reshaped, 1, 0)
E_reshaped.shape
```

```python
p = assoc_dist[mask_excess, 0,]
#p /= np.sum(p)
```

```python
p.shape
```

```python
N_samples = 10_000

rng = np.random.default_rng(seed=42)

bootstrapped_E = rng.choice(np.swapaxes(fit["E"][0][:, mask_excess], 1, 0).flatten(), N_samples, p=p)
```

```python

```

```python
fit._calculate_flux_grid()
```

```python
E_hdi = hdi(bootstrapped_E, 0.68)
```

```python
fig, ax = fit.plot_flux_band(E_power=2.0)
ax.vlines(E_hdi, 1e-11, 1e-10)
```

```python
from arviz_stats import hdi
```

```python
hdi(fit["E"][0, :, 0], 0.68)
```

```python
"L" in fit.keys().keys()
```

```python
fit._fit_nex
```

```python
fig,axs = fit.plot_trace_and_priors(["L_ind", "Nex_src", "Nex_bg", "src_index"])
fig.tight_layout()
```

```python
fit["E"].shape
```

```python
events = Events.from_file("txs.h5")
```

```python
events.N
```

```python
fit._priors.to_dict()["src_index"].pdf(3.0)
```

```python
fit.priors.src_index.mu
```

```python
fit.keys()
```

```python
fit.plot_trace_and_priors(["L_ind"], transform=True)
```

```python
fit.plot_trace(["E"], transform=lambda x: np.log10(x), vector_indices=[0, 3, 61])
```

```python
fit.plot_trace_and_priors(["src_index"])
```

```python
fit._get_kde("src_index")
```

```python
np.atleast_2d(axs)[0, 1]
```

```python

```
