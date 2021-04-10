---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: hierarchical_nu
    language: python
    name: hierarchical_nu
---

# Demonstrate simulation and fit

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
import corner
import astropy.units as u
```

## Sources

```python
import sys
sys.path.append("../")
```

```python
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
```

First set up the high-level parameters. The parameters defined here are singletons that can be accessed throught the program. The `src_index` and `diff_index` refer to the power law spectral index of the point sources and diffuse background. Currently, all sources share the same index, and the background can also be the same or defined separately. `L` and `F_diff` are used to set the normalisation of the point source and diffuse background spectra, defined at `Enorm`. `Emin` and `Emax` bound the source power law spectra for all sources, and define the energy band over which `L` is calculated. 

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
diff_index = Parameter(2.5, "diff_index", fixed=False, par_range=(1, 4))
L = Parameter(3E47 * (u.erg / u.s), "luminosity", fixed=True, par_range=(0, 1E60))
diffuse_norm = Parameter(2e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(0, np.inf))
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(5E4 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
```

When setting the minimum detected (i.e. reconstructed) energy, there are a few options. If fitting one event type (ie. tracks or cascades), just use `Emin_det`. This is also fine if you are fitting both event types, but want to set the same minimum detected energy. `Emin_det_tracks` and `Emin_det_cascades` are to be used when fitting both event types, but setting different minimum detected energies. 

```python
#Emin_det = Parameter(1E5 * u.GeV, "Emin_det", fixed=True)

Emin_det_tracks = Parameter(1e5 * u.GeV, "Emin_det_tracks", fixed=True)
Emin_det_cascades = Parameter(6e4 * u.GeV, "Emin_det_cascades", fixed=True)
```

Next, we use these high-level parameters to define sources. This can be done for either individual sources, or a list loaded from a file. For now we just work with a single point source. There are functions to add the different background components.

```python
# Single PS for testing and usual components
point_source = PointSource.make_powerlaw_source("test", np.deg2rad(5)*u.rad,
                                                np.pi*u.rad, 
                                                L, src_index, 0.43, Emin, Emax)

# Multiple sources from file
#source_file = "my_source_file.h5"
#point_sources = PointSource.make_powerlaw_sources_from_file(source_file, L, 
#                                                            index, Emin, Emax)

my_sources = Sources()
#my_sources.add(point_sources)
#my_sources.select_below_redshift(0.8)
my_sources.add(point_source)

# auto diffuse component 
my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index) 
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

In order to go from sources to a simulation, we need to specify an observation time and a detector model. The detector model defines the effective area, energy resolution and angular resolution to be simulated. The currently implemented options are `NorthernTracksDetectorModel`, `CascadesDetectorModel` and `IceCubeDetectorModel`. The `IceCubeDetectorModel` is really a wrapper around the models for tracks and cascades, for an easy interface. The models should be used in conjunction with the correct `Edet_min`, as described above.

```python
obs_time = 10 * u.year
#sim = Simulation(my_sources, CascadesDetectorModel, obs_time)
#sim = Simulation(my_sources, NorthernTracksDetectorModel, obs_time)
sim = Simulation(my_sources, IceCubeDetectorModel, obs_time)
```

Below are shown all the necessary steps to set up and run a simulation for clarity. There is also the handy sim.setup_and_run() option which calls everything.

```python
sim.precomputation()
sim.generate_stan_code()
sim.compile_stan_code()
sim.run(verbose=True, seed=42)
sim.save("output/test_sim_file.h5")
```

We can visualise the simulation results to check that nothing weird is happening. For the default settings in this notebook, you should see around ~67 simulated events with a clear source in the centre of the sky. The source events are shown in red, diffuse background in blue at atmospheric events in green. The size of the event circles reflects their angular uncertainty (for track events this is exaggerated to make them visible).

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

We can start setting up the fit by loading the events from the output of our simulation. This file only contains the information we would have in a realistic data scenario (energies, directions, uncertainties, event types). We also need to specify the observation time and detector model for the fit, as for the simulation. Please make sure you are using the same ones in both for sensible results!

```python
events = Events.from_file("output/test_sim_file.h5")
obs_time = 10 * u.year
```

```python
#fit = StanFit(my_sources, CascadesDetectorModel, events, obs_time)
#fit = StanFit(my_sources, NorthernTracksDetectorModel, events, obs_time)
fit = StanFit(my_sources, IceCubeDetectorModel, events, obs_time)
```

Similar to the simulation, here are the steps to set up and run a fit. There is also a `fit.setup_and_run()` method available for tidier code. Here, lets run the fit for 2000 samples on a single chain (default setting). This takes around 15 min on one core.

```python
fit.precomputation()
fit.generate_stan_code()
fit.compile_stan_code()
fit.run(show_progress=True, seed=42)
```

Some methods are included for basic plots, but the `fit._fit_output` is a `CmdStanMCMC` object that can be passed to `arviz` for fancier options.

```python
fit.plot_trace()
```

```python
fit.save("output/test.h5")
```

We can check the results of the fit against the known true values from the above simulation. The `SimInfo` class pulls the interesting information out of our saved simulation for this purpose. 

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
