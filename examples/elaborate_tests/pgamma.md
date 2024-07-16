---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: hi_nu
    language: python
    name: python3
---

```python
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.flux_model import LogParabolaSpectrum, PowerLawSpectrum, PGammaSpectrum
from hierarchical_nu.backend.stan_generator import StanGenerator

from cmdstanpy import CmdStanModel
```

```python
with StanGenerator() as gc:
    PGammaSpectrum.make_stan_utility_func(False, False, True)
    PGammaSpectrum.make_stan_flux_conv_func("flux_conv", False, False, True)
    PGammaSpectrum.make_stan_lpdf_func("src_spectrum_lpdf", False, False, True)
    code = gc.generate()
print(code)
```

```python
model = CmdStanModel(stan_file="code.stan")
```

```python
N = 100
Emin = 1e2
Emax = 1e8
E0 = 1e6
energy = np.geomspace(Emin, Emax, N)
data = {
    "N": N,
    "energy": energy,
    "E0": E0,
    "E_min": Emin,
    "E_max": Emax,
}
samples = model.sample(data=data, fixed_param=True, iter_sampling=1, iter_warmup=1, show_console=True)
```

```python
plt.plot(np.geomspace(Emin, Emax, N), np.exp(samples.stan_variable("lpdf").squeeze()))
plt.xscale("log")
plt.yscale("log")

```

```python
Parameter.clear_registry()
index = Parameter(0., "src_index")
alpha = Parameter(0., "alpha")
beta = Parameter(0.7, "beta")
E0 = Parameter(1e6 * u.GeV, "E0_src", fixed=True)

norm = Parameter(1 / u.GeV / u.s / u.m**2, "norm", fixed=True)

Emin = Parameter(1e2 * u.GeV, "Emin_src", fixed=True)
Emax = Parameter(1e8 * u.GeV, "Emax_src", fixed=True)

pgamma = PGammaSpectrum(norm, E0, Emin.value, Emax.value)
Ebreak = Parameter(pgamma.Ebreak, "Ebreak")
#print(Ebreak.value)
logp = LogParabolaSpectrum(norm, E0, alpha, beta, Emin.value, Emax.value)
pl_norm = Parameter(logp(Ebreak.value), "pl_norm")
pl = PowerLawSpectrum(pl_norm, Ebreak.value, index, Emin.value, Ebreak.value)
```

```python
pgamma.integral(*pgamma.energy_bounds), pgamma.total_flux_density.to_value(u.GeV / u.m**2 / u.s)
```

```python
logp.parameters
```

```python
E = np.geomspace(1e3, 1e9, 1_000) << u.GeV
plt.plot(E.value, logp(E).value)
plt.xscale("log")
plt.yscale("log")
```

```python
Ebreak
```

```python
pgamma.Ebreak
```

```python
samples.stan_variable("conv"), pgamma.flux_conv()
```

```python
flux_convs = np.zeros(energy.size) << 1 / u.GeV
E0.fixed = False
for c, e in enumerate(energy):
    E0.value = e * u.GeV
    flux_convs[c] = pgamma.flux_conv()
E0.reset()
```

```python
plt.plot(energy, flux_convs, label="python")
plt.plot(energy, samples.stan_variable("conv").squeeze(), label="stan")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("peak energy [GeV]")
plt.ylabel("flux conversion factor [1 / GeV]")
plt.legend()
plt.savefig("flux_conv_Pgamma_spectrum.pdf")
```

```python
E = np.geomspace(1e3, 1e9, 1_000) << u.GeV
flux = pgamma(E)
plt.plot(E.value, np.log(flux.value / pgamma.integral(Emin.value, Emax.value).to_value(1 / u.m**2 / u.s)))

flux = pl(E)
plt.plot(E.value, np.log(flux.value))

flux = logp(E)
plt.plot(E.value, np.log(flux.value))

plt.xscale("log")
```

```python
plt.plot(E.value, pgamma.pdf(E, Emin.value, Emax.value).value)
plt.xscale("log")
plt.yscale("log")
```

```python
pl.integral(Emin.value, Ebreak.value) + logp.integral(Ebreak.value, Emax.value)
```

```python
pgamma.integral(Emin.value, Ebreak.value) + pgamma.integral(Ebreak.value, Emax.value), pgamma.integral(Emin.value, Emax.value)
```

```python
pgamma.total_flux_density, logp.total_flux_density + pl.total_flux_density
```

```python
pgamma.flux_conv()
```

```python
pgamma(pgamma.Ebreak)
```

```python
logp.flux_conv()
```

$d/dx \left(\frac{E}{E_0}\right)^{-\alpha-\beta \log{(E / E0)}} = (- \alpha -\beta\log{(E/E_0)})$

```python
def logparabola(E, N0, alpha, beta, E0):
    return N0 * np.power(E / E0, -alpha - beta * np.log(E / E0))

def logp_index(E, E0, alpha, beta):
    return alpha + beta * np.log(E / E0) - 
def powerlaw(E, N0, alpha, E0):
    return N0 * np.power(E / E0, -alpha)

def pgamma(E, N0, alpha, beta, E0, Ebreak):
    """
    N0, alpha, beta, E0 are parameters of logparabola
    Ebreak is energy where the spectrum switches definition
    """

    # find index and normalisation at break energy
    index_break = logp_index(Ebreak, E0, alpha, beta)
    val_break = logparabola(Ebreak, N0, alpha, beta, E0)
    
    E = np.atleast_1d(E)

    output = np.zeros_like(E)
    output[E<=Ebreak] = powerlaw(E[E<=Ebreak], val_break, index_break, Ebreak)
    output[E>Ebreak] = logparabola(E[E>Ebreak], N0, alpha, beta, E0)
    return output
```

```python
N0 = 1e-10
alpha = 2.
beta = 0.1
E0 = 1e6
Ebreak = 1e4

E = np.geomspace(1e3, 1e8, 1_000)

plt.plot(E, logparabola(E, N0, alpha, beta, E0))
plt.plot(E, powerlaw(E, N0, alpha, E0))
index_pl = logp_index(Ebreak, E0, alpha, beta)
plt.plot(E, powerlaw(E, N0, index_pl, E0))
plt.plot(E, pgamma(E, N0, alpha, beta, E0, Ebreak))
plt.xscale("log")
plt.yscale("log")
plt.grid()

```

```python
plt.plot(E, logp_index(E, E0, alpha, beta))
plt.xscale("log")
plt.grid()
```

```python
logp_index(1e6, E0, alpha, beta)
```

```python

```
