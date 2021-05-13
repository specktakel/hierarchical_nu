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

# Low energy threshold

Debugging issues with Emin < 5e4 and low Emin_det

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
import corner
import astropy.units as u
```

```python
import sys
sys.path.append("../")
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
diff_index = Parameter(2.0, "diff_index", fixed=False, par_range=(1, 4))
L = Parameter(1E47 * (u.erg / u.s), "luminosity", fixed=True, par_range=(0, 1E60))
diffuse_norm = Parameter(1e-13 /u.GeV/u.m**2/u.s, "diffuse_norm", fixed=True, 
                         par_range=(0, np.inf))
Enorm = Parameter(1E5 * u.GeV, "Enorm", fixed=True)
Emin = Parameter(1E4 * u.GeV, "Emin", fixed=True)
Emax = Parameter(1E8 * u.GeV, "Emax", fixed=True)
```

```python
Emin_det = Parameter(5e4 * u.GeV, "Emin_det", fixed=True)

#Emin_det_tracks = Parameter(1e4 * u.GeV, "Emin_det_tracks", fixed=True)
#Emin_det_cascades = Parameter(1e4 * u.GeV, "Emin_det_cascades", fixed=True)
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
#my_sources.add_atmospheric_component() # auto atmo component
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

```python
obs_time = 10 * u.year
#sim = Simulation(my_sources, CascadesDetectorModel, obs_time)
sim = Simulation(my_sources, NorthernTracksDetectorModel, obs_time)
#sim = Simulation(my_sources, IceCubeDetectorModel, obs_time)
```

```python
sim.precomputation()
sim.generate_stan_code()
sim.compile_stan_code()
sim.run(verbose=True, seed=np.random.randint(100, 10000))
sim.save("output/test_sim_file.h5")
```

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
from hierarchical_nu.simulation import SimInfo
```

```python
events = Events.from_file("output/test_sim_file.h5")
obs_time = 10 * u.year
```

```python
#fit = StanFit(my_sources, CascadesDetectorModel, events, obs_time)
fit = StanFit(my_sources, NorthernTracksDetectorModel, events, obs_time)
#fit = StanFit(my_sources, IceCubeDetectorModel, events, obs_time)
```

```python
fit.precomputation()
fit.generate_stan_code()
#fit.set_stan_filename("../hierarchical_nu/stan/model_test.stan")
```

```python
fit.compile_stan_code()
fit.run(show_progress=True, seed=42, chains=1)
```

```python
fit.plot_trace()
```

```python
fit.save("output/test.h5")
```

```python
sim_info = SimInfo.from_file("output/test_sim_file.h5")
fig = fit.corner_plot(truths=sim_info.truths)
```

```python
fit.check_classification(sim_info.outputs)
```

## Debugging

```python
lp = fit._fit_output.sampler_variables()["lp__"]
```

```python
fig, ax = plt.subplots()
ax.hist(lp)
ax.hist(lp_broken);
```

```python
lp_broken = lp
```

```python
fit_inputs = fit._fit_inputs
```

```python
fig, ax = plt.subplots()
ax.plot(fit_inputs["E_grid"], fit_inputs["Pdet_grid"][0])
ax.plot(fit_inputs["E_grid"], fit_inputs["Pdet_grid"][1])
ax.set_xscale("log")
ax.set_xlim(1e3, 1e8)
#ax.set_ylim(0, 1e4)
ax.set_yscale("log")
```

```python
fig, ax = plt.subplots()
ax.hist(fit._fit_output.stan_variable("Nex"));
ax.hist(Nex_broken)
```

```python
Nex_broken = fit._fit_output.stan_variable("Nex")
```

```python
fig, ax = plt.subplots()
ax.hist(np.log10(fit._fit_output.stan_variable("Fsrc")));
ax.axvline(np.log10(point_source.flux_model.total_flux_int.value), color='k')
```

```python
fig, ax = plt.subplots()
ax.plot(fit_inputs["src_index_grid"], fit_inputs["integral_grid"][0])
ax.plot(fit_inputs["diff_index_grid"], fit_inputs["integral_grid"][1])
ax.set_yscale("log")
```

```python
from hierarchical_nu.source.flux_model import flux_conv_
```

```python
fig, ax = plt.subplots()
index_grid = np.linspace(1, 4)
ax.plot(index_grid, [flux_conv_(_, 1e5, 1e8) for _ in index_grid])
ax.plot(index_grid, [flux_conv_(_, 5e4, 1e8) for _ in index_grid])
ax.plot(index_grid, [flux_conv_(_, 1e4, 1e8) for _ in index_grid])
ax.set_yscale("log")
```

## Check energy resolution

```python
from cmdstanpy import CmdStanModel

from hierarchical_nu.detector.northern_tracks import NorthernTracksDetectorModel
from hierarchical_nu.backend.stan_generator import (
    GeneratedQuantitiesContext,
    DataContext,
    FunctionsContext,
    Include,
    ForLoopContext,
    StanFileGenerator,
)
from hierarchical_nu.backend.variable_definitions import (
    ForwardVariableDef,
    ForwardArrayDef,
)
from hierarchical_nu.backend.expression import StringExpression
from hierarchical_nu.backend.parameterizations import DistributionMode
from hierarchical_nu.stan_interface import STAN_PATH
```

```python
file_name = "output/test_dm"
e_true_name = "e_true"
e_reco_name = "e_recos"
true_dir_name = "true_dirs"
reco_zenith_name = "reco_zeniths"
with StanFileGenerator(file_name) as code_gen:

    with FunctionsContext():

        _ = Include("interpolation.stan")
        _ = Include("utils.stan")
        _ = Include("vMF.stan")

    with DataContext():

        array_length = ForwardVariableDef("n", "int")
        array_length_str = ["[", array_length, "]"]

        e_true = ForwardVariableDef(e_true_name, "real")
        e_recos = ForwardArrayDef(e_reco_name, "real", array_length_str)

    with GeneratedQuantitiesContext():

        ntd = NorthernTracksDetectorModel()

        e_res_result = ForwardArrayDef("e_res", "real", array_length_str)

        with ForLoopContext(1, array_length, "i") as i:

            e_res_result[i] << ntd.energy_resolution(e_true, e_recos[i])
                
    code_gen.generate_single_file()
```

```python
stanc_options = {"include_paths": ["/Users/fran/projects/hierarchical_nu/hierarchical_nu/stan"]}
stan_model = CmdStanModel(
    stan_file=code_gen.filename,
    stanc_options=stanc_options,
)

n = 100
e_reco = np.logspace(2, 9, n)
e_true =6e3
reco_zeniths = np.radians(np.linspace(85, 95, n))
thetas = np.pi - np.radians(np.linspace(85, 180, n, endpoint=False))
true_dir = np.asarray([np.sin(thetas), np.zeros_like(thetas), np.cos(thetas)]).T

data = {
    e_true_name: e_true,
    e_reco_name: e_reco,
    "n": n,
}

output = stan_model.sample(
    data=data,
    iter_sampling=1,
    chains=1,
    fixed_param=True,
    seed=1,
)

e_res = output.stan_variable("e_res")
```

```python
fig, ax = plt.subplots()
ax.plot(e_reco, np.exp(output.stan_variable("e_res").squeeze()))
ax.set_xscale("log")
ax.axvline(e_true)
```

```python

```

```python

```
