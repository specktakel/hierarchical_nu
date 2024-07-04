---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
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
from astropy.coordinates import SkyCoord
```

```python
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources, PointSource
from hierarchical_nu.utils.roi import RectangularROI
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
Emin_det = Parameter(5e4 * u.GeV, "Emin_det", fixed=True)

#Emin_det_northern_tracks = Parameter(1e5 * u.GeV, "Emin_det_tracks", fixed=True)
#Emin_det_cascades = Parameter(6e4 * u.GeV, "Emin_det_cascades", fixed=True)
```

```python
# Single PS for testing and usual components
ra = np.pi*u.rad
dec = np.deg2rad(5)*u.rad
width = np.deg2rad(10) * u.rad
point_source = PointSource.make_powerlaw_source(
    "test", dec, ra, L, src_index, z, Emin_src, Emax_src
)

my_sources = Sources()
my_sources.add(point_source)

# auto diffuse component 
my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff) 

```

```python
roi = RectangularROI(RA_min=ra-width, RA_max=ra+width, DEC_min=dec-width, DEC_max=dec+width)
```

## Test particle

```python
from hierarchical_nu.events import Events
from hierarchical_nu.detector.icecube import IC86_II
```

```python
ra = np.array([185]) * u.deg
dec = np.array([7]) * u.deg
coord = SkyCoord(ra, dec, frame="icrs")

event = Events(energies=np.array([5.1e4]) * u.GeV, 
               coords=coord, 
               types=np.array([IC86_II.S]), 
               ang_errs=np.array([5]) * u.deg,
               mjd=[99])
```

```python
fit.get_src_position(0).separation(coord).deg
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
from hierarchical_nu.fit import StanFit
```

```python
obs_time = 1 * u.year
fit = StanFit(my_sources, IC86_II, event, {IC86_II: obs_time})
fit.setup_and_run()
```

```python
fit.plot_trace("E", transform=lambda x: np.log10(x))
```

```python
fit._get_event_classifications()
```

```python
fit.plot_energy_and_roi(radius=8*u.deg)
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
ax.set_xlabel("association probability")
ax.set_ylabel("pdf")
```

```python
fit.corner_plot();
```

```python

```

```python

```
