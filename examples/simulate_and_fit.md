---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: bayes
    language: python
    name: bayes
---

# Demonstrate simulation and fit

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

## Sources

```python
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
```

```python
# define high-level parameters
Parameter.clear_registry()
index = Parameter(2.0, "index", fixed=False, par_range=(1.0, 4))
L = Parameter(3E47 * (u.erg / u.s), "luminosity", fixed=True, par_range=(0, 1E60))
diffuse_norm = Parameter(2e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(0, np.inf))
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(5E4 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
#Emin_det = Parameter(1E5 * u.GeV, "Emin_det", fixed=True)

Emin_det_tracks = Parameter(1e5 * u.GeV, "Emin_det_tracks", fixed=True)
Emin_det_cascades = Parameter(6e4 * u.GeV, "Emin_det_cascades", fixed=True)

# Single PS for testing and usual components
point_source = PointSource.make_powerlaw_source("test", np.deg2rad(5)*u.rad,
                                                np.pi*u.rad, 
                                                L, index, 0.43, Emin, Emax)

# Multiple sources from file
#source_file = "../dev/statistical_model/data/test_SFR_pop.h5"
#point_sources = PointSource.make_powerlaw_sources_from_file(source_file, L, 
#                                                            index, Emin, Emax)

my_sources = Sources()
#my_sources.add(point_sources)
#my_sources.select_below_redshift(0.8)
my_sources.add(point_source)

my_sources.add_diffuse_component(diffuse_norm, Enorm.value) # auto diffuse component 
my_sources.add_atmospheric_component() # auto atmo component
```

```python
my_sources.associated_fraction()
```

## Simulation

```python
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.detector.cascades import CascadesDetectorModel 
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
```

```python
obs_time = 10 * u.year
#sim = Simulation(my_sources, CascadesDetectorModel, obs_time)
#sim = Simulation(my_sources, NorthernTracksDetectorModel, obs_time)
sim = Simulation(my_sources, IceCubeDetectorModel, obs_time)
```

```python
sim.precomputation()
sim.generate_stan_code()
sim.compile_stan_code()
sim.run(verbose=True, seed=42)
sim.save("output/test_sim_file.h5")
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
```

```python
events = Events.from_file("output/test_sim_file.h5")
obs_time = 10 * u.year
```

```python
#fit = StanFit(my_sources, CascadesDetectorModel, events, obs_time)
#fit = StanFit(my_sources, NorthernTracksDetectorModel, events, obs_time)
fit = StanFit(my_sources, IceCubeDetectorModel, events, obs_time)
```

```python
fit.precomputation()
fit.generate_stan_code()
fit.compile_stan_code()
fit.run(show_progress=True, seed=42)
```

```python
fit.plot_trace()
```

```python
fit.save("output/test.h5")
```

We can check the results of the fit against the known true values from the above simulation.

```python
from hierarchical_nu.simulation import SimInfo
```

```python
sim_info = SimInfo.from_file("output/test_sim_file.h5")
fig = fit.corner_plot(truths=sim_info.truths)
```

Similarly, we can use the simulation info to check the classification of individual events. We shouldn't be concerned if things are slighty off, particularly between the two background components. 

```python
fit.check_classification(sim_info.outputs)
```

```python

```
