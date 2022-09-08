---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
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
from astropy.coordinates import SkyCoord
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

```

## Test particle

```python
from hierarchical_nu.events import Events, TRACKS
```

```python
ra = np.array([187]) * u.deg
dec = np.array([9]) * u.deg
coord = SkyCoord(ra, dec, frame="icrs")

event = Events(energies=np.array([5.1e4]) * u.GeV, 
               coords=coord, 
               types=np.array([TRACKS]), 
               ang_errs=np.array([5]) * u.deg)
```

## Plot inputs

```python
import ligo.skymap.plot
from hierarchical_nu.utils.plotting import SphericalCircle
```

```python
event.coords.representation_type = "spherical"

fig, ax = plt.subplots(subplot_kw={"projection" : "astro degrees mollweide"})
fig.set_size_inches((7, 5))
ax.scatter(point_source.ra.to(u.deg), point_source.dec.to(u.deg),
           transform=ax.get_transform("icrs"), color='k', label="point source")
circle = SphericalCircle((event.coords.ra, event.coords.dec),  event.ang_errs[0], 
                         transform=ax.get_transform("icrs"), label="event", alpha=0.5)
ax.add_patch(circle)
ax.legend()
```

## Fit

```python
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
```

```python
obs_time = 1 * u.year
fit = StanFit(my_sources, NorthernTracksDetectorModel, event, obs_time)
fit.setup_and_run()
```

## P(label=src | data)

```python
log_probs = fit._fit_output.stan_variable("lp").transpose(1, 2, 0)
normalized_probs = np.zeros_like(log_probs)
for j, lp in enumerate(log_probs):
    nlp = []
    for i in  [0,1]:
        nlp.append(np.exp(log_probs[j,i]))
    norm = sum(nlp)
    normalized_probs[j] = np.exp(log_probs[j])/norm
```

```python
fig, ax = plt.subplots()
bins = np.linspace(0, 1)
ax.hist(normalized_probs[0][0], label="point source", alpha=0.5, bins=bins, density=True)
ax.hist(normalized_probs[0][1], label="diffuse bg", alpha=0.5, bins=bins, density=True)
ax.legend()
```

```python
fit.corner_plot();
```

```python

```

```python

```
