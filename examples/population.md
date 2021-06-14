---
jupyter:
  jupytext:
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

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
from astropy import units as u
```

## Simulate a simple source population

Using the popsynth package, it is easy to simulate realistic cosmological populations. We can then use this as a starting point for our usual simulation of neutrinos via Stan.

```python
import popsynth
from popsynth.distributions.delta_distribution import DeltaDistribution
from popsynth.distributions.pareto_distribution import ParetoDistribution
from popsynth.distributions.cosmological_distribution import SFRDistribution
from popsynth.selection_probability.flux_selectors import HardFluxSelection
from popsynth.aux_samplers.delta_aux_sampler import DeltaAuxSampler
from popsynth.aux_samplers.normal_aux_sampler import NormalAuxSampler
from popsynth.population_synth import PopulationSynth
```

First, choose a luminosity function and cosmological evolution.

```python
# All same luminosity
#lf = DeltaDistribution()
#lf.Lp = 1e47 # erg s^-1

# Power-law LF
lf = ParetoDistribution()
lf.Lmin = 3e46 # erg s^-1
lf.alpha = 1

# SFR-like distribution
sd = SFRDistribution()
sd.r0 = 10 # Gpc^-3
sd.a = 0.0
sd.rise = 1.0
sd.decay = 3.0
sd.peak = 1.0

# Plot the LF and SFR-like distribution
z = np.linspace(0, 5)
L = np.geomspace(lf.Lmin, 1e50)
fig, ax = plt.subplots(1, 2)
fig.set_size_inches((15, 5))
ax[0].plot(L, lf.phi(L))
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("L [erg s^-1]")
ax[0].set_ylabel("phi(L)")
ax[1].plot(z, sd.dNdV(z))
ax[1].set_xlabel("z")
ax[1].set_ylabel("dN/dV [Gpc^-3]")
```

```python
# Make a popsynth object using the luminosity function and evolution
pop_synth = PopulationSynth(sd, lf)

# Add a selection on the detected fluxes
flux_select = HardFluxSelection()
flux_select.boundary = 2e-10 # erg s^-1 cm^-2
pop_synth.set_flux_selection(flux_select)

# Add auxiliary sampler for spectral index

# Spectral indices all the same
#index = DeltaAuxSampler(name="spectral_index", observed=False)
#index.xp = 2.0

# Spectral indices ~ Gaussian distribution
index = NormalAuxSampler(name="spectral_index", observed=False)
index.mu = 2.0
index.tau = 0.1

pop_synth.add_observed_quantity(index)

pop_synth.display()
```

```python
# Use the population synth to simulate a population
popsynth.update_logging_level("INFO")
population = pop_synth.draw_survey(flux_sigma=0.1)
population.display_fluxes();
```

```python
fig, ax = plt.subplots()
ax.hist(population.spectral_index)
ax.set_xlabel("spectral_index")
```

```python
population.display_distances()
```

## Saving the population

We can save and load both the population synthesis and the sampled population as required.

```python
# Population synthesis
pop_synth.write_to("output/test_pop_synth.yaml")
new_pop_synth = pop_synth.from_file("output/test_pop_synth.yaml")
new_pop_synth.display()
```

```python
# Population
population.writeto("output/test_population.h5")
new_population = population.from_file("output/test_population.h5")
new_population.display_fluxes();
```

## Using a popsynth population to define a Sources object

```python
import h5py
from astropy import units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import PointSource, Sources
```

```python
# Dfine some high-level params
Parameter.clear_registry()

# Define the bounds used to define the luminosity
Emin = Parameter(5E4 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)

# Detection thresholds
Emin_det_tracks = Parameter(1e5 * u.GeV, "Emin_det_tracks", fixed=True)
Emin_det_cascades = Parameter(6e4 * u.GeV, "Emin_det_cascades", fixed=True)

# For diffuse comp
diffuse_norm = Parameter(1e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(0, np.inf))
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
diff_index = Parameter(2.5, "diff_index", fixed=False, par_range=(1, 4))

# NB: if you set L and src_index here, it will overwrite whatever is in the file
#src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
#L = Parameter(1E47 * (u.erg / u.s), "luminosity", fixed=True, par_range=(0, 1E60)*(u.erg/u.s))
```

```python
# Make a list of point sources from the population file
point_src = PointSource.make_powerlaw_sources_from_file("output/test_population.h5",
                                                        lower_energy=Emin,
                                                        upper_energy=Emax)
# NB: This will only load detected objects, unless we pass the use_undetected=True kwarg.
```

```python
# Add on to Sources object 
my_sources = Sources()
my_sources.add(point_src)
my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index) 
my_sources.add_atmospheric_component() # auto atmo component
```

```python
my_sources.associated_fraction()
```

```python
my_sources.sources[0].parameters
```

Now that the population is simulated and loaded, the simluation and fitting proceeds largely the same as before. If there are many point sources without neutrino signals, we may want to try using a shared luminosity or spectral index parameter, or more constrained prior distributions for these parameters in order for the results to be meaningful.


## Simulation

```python
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
```

```python
obs_time = 10 * u.year
sim = Simulation(my_sources, IceCubeDetectorModel, obs_time)
```

```python
sim.precomputation()
sim.generate_stan_code()
sim.compile_stan_code()
```

```python
sim._get_expected_Nnu(sim._get_sim_inputs())
sim._expected_Nnu_per_comp
```

```python
sim.run(verbose=True, seed=42) 
sim.save("output/test_pop_sim_file.h5")
```

```python
sim.show_skymap()
```

```python
sim.show_spectrum()
```


## Fit

```python
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.detector.icecube import IceCubeDetectorModel
from hierarchical_nu.simulation import SimInfo
from hierarchical_nu.priors import Priors, NormalPrior
```

```python
priors = Priors()
atmo_flux = my_sources.atmospheric.flux_model.total_flux_int.value
priors.atmospheric_flux = NormalPrior(mu=atmo_flux, sigma=0.1*atmo_flux)
```

```python
events = Events.from_file("output/test_pop_sim_file.h5")
obs_time = 10 * u.year
fit = StanFit(my_sources, IceCubeDetectorModel, events, obs_time, priors=priors)
```

```python
fit.precomputation()
fit.generate_stan_code()
fit.compile_stan_code()
```

```python
fit.run(show_progress=True, seed=42)
```

```python
fit.save("output/test_pop_fit_file.h5")
```

```python
fit.plot_trace()
```

```python
sim_info = SimInfo.from_file("output/test_pop_sim_file.h5")
```

```python
fig = fit.corner_plot(truths=sim_info.truths)
```

```python
fit.check_classification(sim_info.outputs)
```

```python

```
