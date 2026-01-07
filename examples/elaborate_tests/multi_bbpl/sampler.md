---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: hnu
    language: python
    name: python3
---

```python
from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter, ParScale
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.source.source import Sources, PointSource, DetectorFrame
from hierarchical_nu.detector.icecube import IC86_II
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

from hierarchical_nu.stan.interface import STAN_PATH, STAN_GEN_PATH
```

```python
# define high-level parameters
Parameter.clear_registry()
#src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
#beta_index = Parameter(0.4, "beta_index", fixed=True, par_range=(-.5, 1.0))
E0_src = Parameter(1e5 * u.GeV, "E0_src", fixed=False, par_range=(1e3, 1e8) * u.GeV, scale=ParScale.log)
diff_index = Parameter(2.13, "diff_index", fixed=False, par_range=(1, 4))
L = Parameter(1e47 * (u.erg / u.s), "luminosity", fixed=True, 
              par_range=(0, 1E60) * (u.erg/u.s))
diffuse_norm = Parameter(1e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(0, np.inf))
z = 0.3365
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
point_source = PointSource.make_pgamma_source(
    "test", dec, ra, L, z, E0_src, Emin_src, Emax_src, DetectorFrame,
)
my_sources = Sources()
my_sources.add(point_source)
```

```python
event_types = IC86_II
lifetime = 0.5*u.year
sim = Simulation(my_sources, event_types, lifetime)
```

```python
sim.precomputation()
```

```python
segment = sim._exposure_integral[IC86_II]._envelope_container[0]
```

```python
plt.plot(segment.support, segment.target)
env = [segment(_) for _ in segment.support]
plt.plot(segment.support, env)

plt.xscale("log")
plt.yscale("log")
```

```python
model = CmdStanModel(stan_file="sampler.stan", stanc_options={"include-paths": [STAN_PATH]})
```

```python
data = {
    "E": np.geomspace(1e2, 1e8, 1000),
    "N_E": 1000,
    "slopes": segment.slopes,
    "weights": segment.weights,
    "breaks": segment.bins,
    "N": segment.N,
    "norms": segment.low_values,
}
samples = model.sample(fixed_param=True, data=data, iter_sampling=1, iter_warmup=0)
```

```python
E = np.geomspace(1e2, 1e8, 1000)
pdf = samples.stan_variable("pdf").squeeze()
plt.plot(E, pdf, label="stan pdf implementation", ls="dashdot")

rng = samples.stan_variable("samples").squeeze()

plt.plot(segment.support, segment.target, label="target")
plt.plot(segment.support, env, label="segmented approx", ls="dotted")
plt.xscale("log")
plt.yscale("log")
plt.legend()
```

```python
plt.hist(rng, bins=np.geomspace(1e2, 1e8, 50), density=True, label="stan samples")
env = np.array([segment(_) for _ in segment.support]) / segment.integrals.sum()
plt.plot(segment.support, env, label="segmented approx")
plt.xscale("log")
plt.yscale("log")
plt.legend()
```

```python

```
