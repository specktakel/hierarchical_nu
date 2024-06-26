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

## Spectral shapes


There are three spectral shapes implemented:
 - power law, defined between Emin_src and Emax_src, returning negative infinity outside this range
 - twice broken power law, defined between Emin_src and Emax_src, continued outside this range by steep flanks of index $\pm 15$
 - logparabola, using alpha, beta and E0_src as parameters, currently defined over range of Emin_src to Emax_src.

```python
from astropy.coordinates import SkyCoord
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter, ParScale
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors
from hierarchical_nu.source.source import Sources, PointSource, DetectorFrame
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior, LuminosityPrior, IndexPrior, FluxPrior, EnergyPrior
from hierarchical_nu.utils.roi import CircularROI
from hierarchical_nu.detector.icecube import IC86_II, IC86_I
from hierarchical_nu.detector.input import mceq
from icecube_tools.utils.data import Uptime
import numpy as np
import ligo.skymap.plot
```

```python
ra = np.deg2rad(77.35) * u.rad
dec = np.deg2rad(5.7) * u.rad
width = np.deg2rad(6) * u.rad
txs = SkyCoord(ra=ra, dec=dec, frame="icrs")
```

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1, 4))
L = Parameter(1e47 * (u.erg / u.s), "luminosity", fixed=True, 
              par_range=(0, 1E60) * (u.erg/u.s))
z = 0.3365
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
Emin_src = Parameter(Emin.value, "Emin_src", fixed=True)
Emax_src = Parameter(Emax.value, "Emax_src", fixed=True)
Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

point_source = PointSource.make_powerlaw_source(
    "test", dec, ra, L, src_index, z, Emin_src, Emax_src, DetectorFrame,
)
```

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.2, "src_index", fixed=False, par_range=(1, 4))
L = Parameter(1e47 * (u.erg / u.s), "luminosity", fixed=True, 
              par_range=(0, 1E60) * (u.erg/u.s))
z = 0.3365
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
Emin_src = Parameter(Emin.value, "Emin_src", fixed=True)
Emax_src = Parameter(Emax.value, "Emax_src", fixed=True)
Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

point_source = PointSource.make_twicebroken_powerlaw_source(
    "test", dec, ra, L, src_index, z, Emin_src, Emax_src, DetectorFrame,
)
```

In the case of a logparabola, we are free to choose two out of three shape parameters to fit.
The remaining shape parameter is considered data and is using the kwarg `fixed=True`.

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.2, "src_index", fixed=True, par_range=(1, 4))
beta_index = Parameter(0.1, "beta_index", fixed=False)
E0_src = Parameter(1e5, "E0_src", fixed=False, par_range=(1e3, 1e8)*u.GeV, scale=ParScale.log)
L = Parameter(1e47 * (u.erg / u.s), "luminosity", fixed=True, 
              par_range=(0, 1E60) * (u.erg/u.s))
z = 0.3365
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1E2 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
Emin_src = Parameter(Emin.value, "Emin_src", fixed=True)
Emax_src = Parameter(Emax.value, "Emax_src", fixed=True)
Emin_diff = Parameter(Emin.value, "Emin_diff", fixed=True)
Emax_diff = Parameter(Emax.value, "Emax_diff", fixed=True)

point_source = PointSource.make_logparabola_source(
    "test", dec, ra, L, src_index, beta_index, z, Emin_src, Emax_src, E0_src, DetectorFrame,
)
```

E0_src is the normalisation energy. Considered as a free parameter it may cover multiple orders of magnitude, hence we pass the kwarg `scale=ParScale.log`. This instructs the precomputation to calculate using a logarithmic grid and also interpolate over log(E0_src) inside stan.

Further, a 2D interpolation is needed for the logparabola, increasing the runtime of both precomputation and fits.

```python

```
