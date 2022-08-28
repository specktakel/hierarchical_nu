---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: hierarchical_nu
    language: python
    name: hierarchical_nu
---

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
import corner
import astropy.units as u
```

```python
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
```

More efficient rejection sampling

```python
from hierarchical_nu.backend.stan_generator import StanGenerator
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
```

```python
with StanGenerator():
    dm = NorthernTracksDetectorModel()
Aeff = dm.effective_area
```

```python
def src_spectrum_rng(alpha, e_low, e_up):
    norm = ((1-alpha)/((e_up**(1-alpha))-(e_low**(1-alpha))))
    u = np.random.uniform(0, 1)
    return (((u * (1-alpha))/norm)+(e_low**(1-alpha)))**(1/(1-alpha))

def src_spectrum_pdf(E, alpha, e_low, e_up):
    if alpha == 1.0:
        N = 1.0 / (np.log(e_up)-np.log(e_low))
    else:
        N = (1.0 - alpha) / (e_up**(1.0-alpha) - e_low**(1.0-alpha))
    return N * E**(alpha*-1)

def bpl_pdf(E, alpha_1, alpha_2, e_low, e_th, e_up):
    I1 = (e_th**(alpha_1 + 1.0) - e_low**(alpha_1 + 1.0)) / (alpha_1 + 1.0)
    I2 = (e_th**(alpha_1 - alpha_2) * (e_up**(alpha_2 + 1.0) - e_th**(alpha_2 + 1.0)) / 
         (alpha_2 + 1.0))
    N = 1.0 / (I1 + I2)
    if E <= e_th and E >= e_low:
        return N * E**alpha_1
    elif E > e_th and E <= e_up:
        return N * e_th**(alpha_1 - alpha_2) * E**alpha_2
    else:
        return 0 
```

```python
alpha = 2.0
e_low = 1e3
e_up = 1e8

E_arr_0 = []
for i in range(10_000):
    E_arr_0.append(src_spectrum_rng(alpha, e_low, e_up))
```

```python
bins = Aeff.tE_bin_edges

fig, ax = plt.subplots()
#ax.hist(E_arr_0, bins=bins, density=True, alpha=0.2)
#ax.plot(bins[:-1], src_spectrum_pdf(bins[:-1], alpha, e_low, e_up))
#for i in range(len(Aeff.eff_area.T)):
#    ax.plot(bins[:-1], Aeff.eff_area.T[i]/np.max(Aeff.eff_area.T))
for i in range(len(Aeff.eff_area.T)):
    ax.plot(bins[:-1], src_spectrum_pdf(bins[:-1], alpha, e_low, e_up) 
            * (Aeff.eff_area.T[i]/max(Aeff.eff_area.T[i])))
ax.plot(bins[:-1], [bpl_pdf(b, -1.5, -3, e_low, 3e6, e_up) for b in bins[:-1]], 
        color="k", linestyle="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-15)
ax.set_xlim(e_low, e_up)
```

 o ## Sources

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
diff_index = Parameter(2.7, "diff_index", fixed=False, par_range=(1, 4))
L = Parameter(5E46 * (u.erg / u.s), "luminosity", fixed=True, 
              par_range=(0, 1E60) * (u.erg/u.s))
diffuse_norm = Parameter(1e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(0, np.inf))
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1E4 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
```

```python
Emin_det = Parameter(5e4 * u.GeV, "Emin_det", fixed=True)

#Emin_det_tracks = Parameter(1e5 * u.GeV, "Emin_det_tracks", fixed=True)
#Emin_det_cascades = Parameter(6e4 * u.GeV, "Emin_det_cascades", fixed=True)
```

```python
# Single PS for testing and usual components
point_source = PointSource.make_powerlaw_source("test", np.deg2rad(5)*u.rad,
                                                np.pi*u.rad, 
                                                L, src_index, 0.4, Emin, Emax)

my_sources = Sources()
my_sources.add(point_source)

# auto diffuse component 
my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index) 
#my_sources.add_atmospheric_component() # auto atmo component
```

```python
my_sources.f_arr()
```

## Simulation

```python
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
```

```python
obs_time = 10 * u.year
sim = Simulation(my_sources, NorthernTracksDetectorModel, obs_time)
```

```python
sim.precomputation()
sim.generate_stan_code()
sim.compile_stan_code()
sim.run(verbose=True, seed=np.random.randint(10000))
sim.save("output/test_sim_file.h5")
```

```python
sim._expected_Nnu_per_comp
```

```python
fig, ax = sim.show_spectrum()
```

```python
fig, ax = sim.show_skymap()
```

## Fit 

```python
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.simulation import SimInfo
```

```python
events = Events.from_file("output/test_sim_file.h5")
obs_time = 10 * u.year
```

```python
fit = StanFit(my_sources, NorthernTracksDetectorModel, events, obs_time)
```

```python
fit.precomputation()
fit.generate_stan_code()
fit.compile_stan_code()
fit.run(show_progress=True, seed=np.random.randint(10000), chains=1)
```

```python
fit.plot_trace()
```

```python
fit.save("output/test.h5")
```

```python
sim_info = SimInfo.from_file("output/test_sim_file.h5")
fig = fit.corner_plot(truths=sim_info.truths)
```

```python
fit.check_classification(sim_info.outputs)
```

```python

```
