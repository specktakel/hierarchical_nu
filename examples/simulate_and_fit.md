---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
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
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
```

First set up the high-level parameters. The parameters defined here are singletons that can be accessed throught the program. The `src_index` and `diff_index` refer to the power law spectral index of the point sources and diffuse background. Currently, all sources share the same index, and the background can also be the same or defined separately. `L` and `F_diff` are used to set the normalisation of the point source and diffuse background spectra, defined at `Enorm`. `Emin` and `Emax` bound the source power law spectra for all sources, and define the energy band over which `L` is calculated. 

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

When setting the minimum detected (i.e. reconstructed) energy, there are a few options. If fitting one event type (ie. tracks or cascades), just use `Emin_det`. This is also fine if you are fitting both event types, but want to set the same minimum detected energy. `Emin_det_tracks` and `Emin_det_cascades` are to be used when fitting both event types, but setting different minimum detected energies. 

```python
Emin_det = Parameter(6e4 * u.GeV, "Emin_det", fixed=True)

#Emin_det_tracks = Parameter(1e5 * u.GeV, "Emin_det_tracks", fixed=True)
#Emin_det_cascades = Parameter(6e4 * u.GeV, "Emin_det_cascades", fixed=True)
```

Next, we use these high-level parameters to define sources. This can be done for either individual sources, or a list loaded from a file. For now we just work with a single point source. There are functions to add the different background components.

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
my_sources.f_arr() # Associated fraction of arrival flux
```

```python
my_sources.f_arr_astro() # As above, excluding atmo
```

## Simulation

```python
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.detector.cascades import CascadesDetectorModel 
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.detector.r2021 import R2021DetectorModel
```

In order to go from sources to a simulation, we need to specify an observation time and a detector model. The detector model defines the effective area, energy resolution and angular resolution to be simulated. The currently implemented options are `NorthernTracksDetectorModel`, `CascadesDetectorModel` and `IceCubeDetectorModel`. The `IceCubeDetectorModel` is really a wrapper around the models for tracks and cascades, for an easy interface. The models should be used in conjunction with the correct `Edet_min`, as described above.

```python
obs_time = 5 * u.year
#sim = Simulation(my_sources, CascadesDetectorModel, obs_time)
#sim = Simulation(my_sources, NorthernTracksDetectorModel, obs_time)
sim = Simulation(my_sources, R2021DetectorModel, obs_time)
```

Below are shown all the necessary steps to set up and run a simulation for clarity. There is also the handy sim.setup_and_run() option which calls everything.

```python
sim.precomputation()
```

To print the number of expected events, call the following methods.

```python
print(sim._get_expected_Nnu(sim._get_sim_inputs()))
print(sim._expected_Nnu_per_comp)
```

```python
sim.generate_stan_code()
sim.compile_stan_code()
sim.run(verbose=True, seed=42) 
sim.save("output/test_sim_file.h5", overwrite=True)
```

For already compiled models that should be re-loaded there is a special method `sim.setup_stan_sim(".stan_files/sim_code")` which takes the filename of the compiled model as sole argument. Source selection should macht up with the compiled model, otherwise cmdstanpy will complain. The `save` method (also the one of the fit) has a keyword `overwrite` which needs to be set to `True`. Otherwise, if a file of the specified name already exsists, an exception is raised.


We can visualise the simulation results to check that nothing weird is happening. For the default settings in this notebook, you should see around ~50 simulated events with a clear source in the centre of the sky. The source events are shown in red, diffuse background in blue at atmospheric events in green. The size of the event circles reflects their angular uncertainty (for track events this is exaggerated to make them visible).

```python
fig, axs = sim.show_spectrum()
# fig, axs = sim.show_spectrum(scale="log")  displays plots with y axis on a log scale
```

## Fit 

```python
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.detector.cascades import CascadesDetectorModel
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior
```

We can start setting up the fit by loading the events from the output of our simulation. This file only contains the information we would have in a realistic data scenario (energies, directions, uncertainties, event types). We also need to specify the observation time and detector model for the fit, as for the simulation. Please make sure you are using the same ones in both for sensible results!

```python
events = Events.from_file("output/test_sim_file.h5")
obs_time = 5 * u.year
```

We can also define priors using the `Priors` interface. Here, we use the default uninformative priors, except for on the atmospheric flux, which we assume to be well known.

```python
priors = Priors()

flux_units = 1 / (u.m**2 * u.s)
atmo_flux = my_sources.atmospheric.flux_model.total_flux_int.to(flux_units).value
priors.atmospheric_flux = LogNormalPrior(mu=np.log(atmo_flux), sigma=0.1)
```

```python
#fit = StanFit(my_sources, CascadesDetectorModel, events, obs_time, priors=priors)
#fit = StanFit(my_sources, NorthernTracksDetectorModel, events, obs_time, priors=priors)
#fit = StanFit(my_sources, IceCubeDetectorModel, events, obs_time, priors=priors)
fit = StanFit(my_sources, R2021DetectorModel, events, obs_time, priors=priors, nshards=10)
# optional keyword nshards=5 or other integer, activates multithreading
```

The kwarg `nshards` accepts integer numbers. Any number greater than 1 will cause the model to be compiled with multithreading. This leads to a different model code, where the data is split up in shards. For each shard one thread is used to calculate the loglikelihood, which in the end is summed up and added to stan's `target`. Using `nshards=1` or not specifying it at all will compile a 'normal' model code without multithreading.


Similar to the simulation, here are the steps to set up and run a fit. There is also a `fit.setup_and_run()` method available for tidier code. Here, lets run the fit for 2000 samples on a single chain (default setting). This takes around 5 min on one core (or roughly one minute on 10 cores).

Sometimes the MCMC needs a little help getting started, for this we can set `inits={"L": 1e50, "src_index": 2.3}` and other model parameters in `fit.run()` with a value to start from.

```python
fit.precomputation()
fit.generate_stan_code()
fit.compile_stan_code()
fit.run(show_progress=True, seed=42)
# fit.setup_stan_fit(".stan_files/model_code")   # will re-load compiled model
```

Some methods are included for basic plots, but the `fit._fit_output` is a `CmdStanMCMC` object that can be passed to `arviz` for fancier options.

```python
axs = fit.plot_trace()
```

We can also overplot the used priors (if there are priors available for the variables) by calling a different method. Both return a list of axes.

```python
axs = fit.plot_trace_and_priors()
```

```python
fit.save("output/test.h5", overwrite=True)
```

We can check the results of the fit against the known true values from the above simulation. The `SimInfo` class pulls the interesting information out of our saved simulation for this purpose. 

```python
from hierarchical_nu.simulation import SimInfo
```

```python
sim_info = SimInfo.from_file("output/test_sim_file.h5")
fig = fit.corner_plot(truths=sim_info.truths)
```

Similarly, we can use the simulation info to check the classification of individual events. We shouldn't be concerned if things are slighty off, particularly between the two background components. The method returns a list of wrongly assosciated events, along with the assumed and correct classification.

```python
wrong, assumed, correct = fit.check_classification(sim_info.outputs)
```

With this information we can update the simulated spectrum and see at which energies, e.g., the background components are competing for events.

```python
fig, axs = sim.show_spectrum()
for es, et, er in zip(sim._sim_output.stan_variable("Esrc")[0][wrong], sim._sim_output.stan_variable("E")[0][wrong], events.energies[wrong].value):
    axs[0].axvline(et, 0.9, 1, color="black")   # at source
    axs[1].axvline(et, 0.9, 1, color="black")   # at detector
    axs[2].axvline(er, 0.9, 1, color="black")   # reconstructed

```

```python
fig, ax = plt.subplots()
ax.hist(fit._fit_output.stan_variable("Nex_src"), alpha=0.5);
ax.hist(fit._fit_output.stan_variable("Nex_diff"), alpha=0.5);
ax.hist(fit._fit_output.stan_variable("Nex_atmo"), alpha=0.5)
ax.axvline(sim._expected_Nnu_per_comp[0], color="blue")
ax.axvline(sim._expected_Nnu_per_comp[1], color="orange")
ax.axvline(sim._expected_Nnu_per_comp[2], color="green")
ax.axvline(np.sum(sim._sim_output.stan_variable("Lambda")[0]==1.), color="blue", ls='--')
ax.axvline(np.sum(sim._sim_output.stan_variable("Lambda")[0]==2.), color="orange", ls='--')
ax.axvline(np.sum(sim._sim_output.stan_variable("Lambda")[0]==3.), color="green", ls='--')
```

```python

```
