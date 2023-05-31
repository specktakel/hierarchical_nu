---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: hi_nu
    language: python
    name: python3
---

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
import corner
import astropy.units as u
```

```python
import sys
sys.path.append("../")
```

```python
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
```

## Sources

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
diff_index = Parameter(3.7, "diff_index", fixed=False, par_range=(1, 4))
L = Parameter(5E46 * (u.erg / u.s), "luminosity", fixed=True, 
              par_range=(0, 1E60) * (u.erg/u.s))
diffuse_norm = Parameter(1e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(0, np.inf))
z = 0.4
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1E4 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
Emin_src = Parameter(Emin.value * (1 + z), "Emin_src", fixed=True)
Emax_src = Parameter(Emax.value * (1 + z), "Emax_src", fixed=True)
Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)
```

```python
#Emin_det = Parameter(6e4 * u.GeV, "Emin_det", fixed=True)

Emin_det_tracks = Parameter(3e4 * u.GeV, "Emin_det_tracks", fixed=True)
Emin_det_cascades = Parameter(6e4 * u.GeV, "Emin_det_cascades", fixed=True)
```

```python
# Single PS for testing and usual components
point_source = PointSource.make_powerlaw_source("test", np.deg2rad(5)*u.rad,
                                                np.pi*u.rad, 
                                                L, src_index, z, Emin_src, Emax_src)

my_sources = Sources()
my_sources.add(point_source)

# auto diffuse component 
my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff) 
my_sources.add_atmospheric_component() # auto atmo component
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
#sim = Simulation(my_sources, NorthernTracksDetectorModel, obs_time)
sim = Simulation(my_sources, IceCubeDetectorModel, obs_time)
```

```python
sim.precomputation()
sim.generate_stan_code()
sim.compile_stan_code()
```

```python
sim.run(verbose=True, seed=1)
```

```python
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

Faster fit coming soon! Currently a bit slow. 

```python
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.simulation import SimInfo
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior
```

```python
events = Events.from_file("output/test_sim_file.h5")
obs_time = 10 * u.year
```

```python
priors = Priors()

flux_units = 1 / (u.m**2 * u.s)
atmo_flux = my_sources.atmospheric.flux_model.total_flux_int.to(flux_units).value
priors.atmospheric_flux = LogNormalPrior(mu=np.log(atmo_flux), sigma=0.1)

#diff_flux = my_sources.diffuse.flux_model.total_flux_int.to(flux_units).value
#priors.diffuse_flux = LogNormalPrior(mu=np.log(diff_flux), sigma=0.2)

#diff_index = my_sources.diffuse.parameters["index"].value
#priors.diff_index = NormalPrior(mu=diff_index, sigma=0.2)
```

```python
#fit = StanFit(my_sources, NorthernTracksDetectorModel, events, obs_time, priors=priors)
fit = StanFit(my_sources, IceCubeDetectorModel, events, obs_time, priors=priors)
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
