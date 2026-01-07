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
from matplotlib import pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from hierarchical_nu.detector.fit_irf_to_data import RateCalculator
from hierarchical_nu.detector.icecube import IC40, IC59, IC79, IC86_I, IC86_II
from hierarchical_nu.detector.r2021 import R2021EffectiveArea, R2021EnergyResolution
from hierarchical_nu.detector.input import mceq
from hierarchical_nu.source.atmospheric_flux import AtmosphericNuMuFlux

```

```python
Parameter.clear_registry()
atmo = AtmosphericNuMuFlux(1e1*u.GeV, 1e9*u.GeV, cache_dir=mceq)
index = Parameter(2.52, "diff_index")
norm = Parameter(1.80e-18 / u.GeV / u.cm**2 / u.s * 4 * np.pi, "norm")
Enorm = Parameter(100 * u.TeV, "Enorm")

pl = PowerLawSpectrum(norm, Enorm.value, index, 1e2 * u.GeV, 1e9 * u.GeV)
diffuse = IsotropicDiffuseBG(pl)
```

```python
calc = RateCalculator(IC86_II, atmo, diffuse, 1)
```

```python
rates = calc.calc_rates(detailed=2)
```

```python
#rates = calc.calc_rates_from_2d_splines(detailed=2)
# calc.plot_detailed_rates(rates);
```

```python
fig, axs = calc.plot_detailed_rates(rates)
```

```python
fig.savefig("ic86_ii_eq_default.pdf")
```

```python
for season in [IC40, IC59, IC79, IC86_I, IC86_II]:
    aeff = R2021EffectiveArea(season=season.P)
    for dec_idx, dec_label in zip([1, 2], ["Equator", "North"]):
        calc = RateCalculator(season, nuflux, dec_idx)
        rates = calc.calc_rates_from_2d_splines(detailed=2)
        fig, axs = calc.plot_detailed_rates(rates)
        axs[0].set_title(f"{season.P}, {dec_label}")
        fig.savefig(f"atmo_rates_{season.P}_{dec_label}_spline.pdf", dpi=150)
```
