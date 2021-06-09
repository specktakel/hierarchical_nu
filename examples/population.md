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
```

## Simulate a simple population of objects

Using the popsynth package, it is easy to simulate realistic cosmological populations. We can then use this as a starting point for our usual simulation of neutrinos via Stan.

```python
import sys
sys.path.append("../../popsynth/")
import popsynth
from popsynth.distributions.delta_distribution import DeltaDistribution
from popsynth.distributions.cosmological_distribution import SFRDistribution
from popsynth.selection_probability.flux_selectors import HardFluxSelection
from popsynth.population_synth import PopulationSynth
```

First, choose a luminosity function and cosmological evolution.

```python
# All same luminosity
lf = DeltaDistribution()
lf.Lp = 1e48 # erg s^-1

# SFR-like distribution
sd = SFRDistribution()
sd.r0 = 10 # Gpc^-3
sd.a = 0.0
sd.rise = 1.0
sd.decay = 3.0
sd.peak = 1.0

# Plot the SFR-like distribution
z = np.linspace(0, 5)
fig, ax = plt.subplots()
ax.plot(z, sd.dNdV(z))
ax.set_xlabel("z")
ax.set_ylabel("dN/dV [Gpc^-3]")
```

```python
# Make a popsynth object using the luminosity function and evolution
pop_synth = PopulationSynth(sd, lf)

# Add a selection on the detected fluxes
flux_select = HardFluxSelection()
flux_select.boundary = 1e-9 # erg s^-1 cm^-2
pop_synth.set_flux_selection(flux_select)

# Add an auxiliary sampler to sample the spectral indices of each source


pop_synth.display()
```

```python
# Use it to simulate a population
popsynth.update_logging_level("INFO")
population = pop_synth.draw_survey(flux_sigma=0.1)
population.display_fluxes();
```



```python

```
