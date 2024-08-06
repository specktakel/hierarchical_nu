---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: hi_nu
    language: python
    name: python3
---

## Analysing experimental data


The 10 year PS data set provides multiple IRFs and events recorded in the particular detector configurations, starting with `IC40` and ending with `IC86_VII`, although the detector configuration did not change after `IC86_II`.

As an example we consider TXS0506+056 during the 2014/2015 flare.

```python
from astropy.coordinates import SkyCoord
from hierarchical_nu.utils.config_parser import ConfigParser
from pathlib import Path
from hierarchical_nu.utils.config import HierarchicalNuConfig
import omegaconf
import astropy.units as u
from hierarchical_nu.source.parameter import Parameter, ParScale
from hierarchical_nu.simulation import Simulation
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors
from hierarchical_nu.source.source import Sources, PointSource, DetectorFrame
from hierarchical_nu.utils.lifetime import LifeTime
from hierarchical_nu.events import Events
from hierarchical_nu.fit import StanFit
from hierarchical_nu.priors import Priors, LogNormalPrior, NormalPrior, LuminosityPrior, IndexPrior, FluxPrior
from hierarchical_nu.utils.roi import CircularROI
from hierarchical_nu.detector.icecube import IC86_II, IC86_I
from hierarchical_nu.detector.input import mceq
from icecube_tools.utils.data import Uptime
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
import ligo.skymap.plot
from scipy.optimize import least_squares, Bounds
from abc import ABCMeta, abstractmethod
```

First, we define the source and fit parameters, as already seen in `simulate_and_fit`. The value of e.g. `src_index` has no meaning in this context. Relevant for the fit is only the existance of the parameter itself. We can, however, use the parameter values to make decisions on which priors to choose by gauging how many events are to be expected by a certain choice of parameters.

```python
# define high-level parameters
Parameter.clear_registry()
src_index = Parameter(2.0, "src_index", fixed=False, par_range=(1, 4))
#beta_index = Parameter(0.4, "beta_index", fixed=True, par_range=(-.5, 1.0))
# E0_src = Parameter(1e5 * u.GeV, "E0_src", fixed=False, par_range=(1e3, 1e8) * u.GeV, scale=ParScale.log)
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
#point_source = PointSource.make_pgamma_source(
#    "test", dec, ra, L, z, E0_src, Emin_src, Emax_src, DetectorFrame,
#)
#
# point_source = PointSource.make_logparabola_source(
#     "test", dec, ra, L, src_index, beta_index, z, Emin_src, Emax_src, E0_src, DetectorFrame
# )

point_source = PointSource.make_twicebroken_powerlaw_source(
    "test", dec, ra, L, src_index, z, Emin_src, Emax_src, DetectorFrame,
)
my_sources = Sources()
my_sources.add(point_source)
#my_sources.add_diffuse_component(diffuse_norm, Enorm.value, diff_index, Emin_diff, Emax_diff) 
#my_sources.add_atmospheric_component(cache_dir=mceq)
```

We now need to decide on the time period of observation (start and end times in MJD). The detector lifetime is automatically calculated from the "good time intervals" provided in the data release. Event selection respects the start and end times with which an ROI is instanciated.

From the [paper](https://arxiv.org/pdf/2101.09836.pdf) accompanying the data release, Table IV, we can guesstimate (Tmin, Tmax) = (56917, 57113). This is not the result of a flare analysis, but rather to include all contributing events.

Together with the source location we create an ROI.

```python
MJD_min=56917
MJD_max=57113

roi = CircularROI(txs, 5 * u.deg, MJD_min=MJD_min, MJD_max=MJD_max, apply_roi=True)
```

A `LifeTime` instance computes the detector lifetime in the chosen time range.

```python
lt = LifeTime()
lifetime = lt.lifetime_from_mjd(MJD_min, MJD_max)
print(lifetime)
```

Returned is dictionary with event type as key and lifetime in years as value. We can re-use this return value to get a list of event types as input for the fit.

```python
event_types = list(lifetime.keys())
```

```python
events = Events.from_ev_file(
    *event_types)
print(events.N)
```

We take a small detour to setup a simulation. Through `sim._get_expected_Nnu` we are able to calculate the expected number of events.
Due to a mismatch between data and MC it is currently sensible to set a prior on the atmospheric flux that will account for the selected number of events, rather than the MCEq-simulated flux value.

Assuming all data is background, we rescale the atmospheric flux to produce the number of events in the data sample. Data taking is a Poisson counts experiment, meaning that $\sqrt{N}$ is the standard deviation of $N$.

```python
sim = Simulation(my_sources, event_types, lifetime, N={IC86_II: [10]})
```

```python
sim.precomputation()
print(sim._get_expected_Nnu(sim._get_sim_inputs()))
print(sim._expected_Nnu_per_comp)
```

```python
Emin = 1e2
Emax = 1e8
logEmin = np.log10(Emin)
logEmax = np.log10(Emax)
E = np.geomspace(Emin, Emax, 1_000)
N = 5

# Define width in decadic unit, half an order of magnitude
width = 1
# get target function
f = sim._exposure_integral[IC86_II]._f_values[0].to_value(u.m**2)
logf = np.log10(f)
logx = np.log10(E)
g = sim._exposure_integral[IC86_II]._g_values[0]
# Find position of max of target function
target_max = f.max()
target_max_point = E[np.argmax(f).squeeze()]
# create break points with max of target as bin center somewhere
middle = np.log10(target_max_point)
breaks = [middle - width / 2, middle + width / 2]
print(breaks)
if breaks[0] < logEmin:
    breaks[0] = logEmin
if breaks[1] > logEmax:
    breaks[1] = logEmax

if breaks[1] < logEmax:
    while True:
        proposal = breaks[-1] + width
        # print(proposal)
        if proposal > np.log10(Emax):
            proposal = np.log10(Emax)
        breaks.append(proposal)

        if proposal == np.log10(Emax):
            break

    
if breaks[0] > logEmin:
    while True:
        proposal = breaks[0] - width
        if proposal < np.log10(Emin):
            proposal = np.log10(Emin)
        breaks.insert(0, proposal)
        if proposal == np.log10(Emin):
            break
    breaks = np.power(10, breaks)

print(breaks)
```

```python
class PL:
    def __init__(self, slope, xmin, xmax, low=True, val=None):
        pass


class SegmentedApprox(metaclass=ABCMeta):
    def __init__(
            self, 
            target,
            support,
            bins,
    ):
        self.target = target
        self.support = support

        self.target_max = np.max(self.target)
        self.support_max = support[np.argmax(self.target)]

        self.log_target = np.log10(self.target)
        self.log_support = np.log10(self.support)

        self.xmin = np.min(support)
        self.xmax = np.max(support)

        self.bins = bins
        self.slopes = []

        self._segmented_functions = [lambda x: -1 for _ in range(len(bins)-1)]

        self.diff = 0.02

    def target_log_approx(self, x):
        return np.power(10., np.interp(np.log10(x), self.log_support, self.log_target))
    
    def segment_factory(self, slope, logxmin, logxmax, val, low=True):
        xmin = np.power(10, logxmin)
        xmax = np.power(10, logxmax)
        def func(x):
            if low:
                x0 = xmin
            else:
                x0 = xmax
            return np.power(x / x0, slope) * val
        return func
        
    def init_slope(self, logxmin, logxmax):
        return np.log10(self.target_log_approx(np.power(10, logxmax)) / self.target_log_approx(np.power(10, logxmin))) / (logxmax - logxmin)
    
    def _fit_segment(self, xmin, xmax, low=True, val=None):
        self._trial_functions = []
        if low and val is None:
            val = self.__call__(xmin)
        elif not low and val is None:
            val = self.__call__(xmax)

        print(val)
        print(xmin, xmax)
        logxmin = np.log10(xmin)
        logxmax = np.log10(xmax)

        print("min/max", logxmin, logxmax)
        
        support = np.geomspace(xmin, xmax)
        
        #Propose first function
        slope = self.init_slope(logxmin, logxmax)
        print("init slope", slope)
        function = self.segment_factory(slope, logxmin, logxmax, val)

        diff = function(support) - self.target_log_approx(support)

        if np.any(diff < 0.) and low:
            step = self.diff
            print("under")
        elif low:
            step = - self.diff
            print("over")
        elif np.any(diff < 0.) and not low:
            step = - self.dff
        elif not low:
            step = self.diff
        print(step)
        for i in range(100):
            new_slope = slope + step
            new_function = self.segment_factory(new_slope, logxmin, logxmax, val)
            # self._trial_functions.append(new_function)
            negative = np.any(new_function(support) - self.target_log_approx(support) < 0.)
            #print(negative)
            if step > 0. and negative:
                slope = new_slope
                continue
            elif step > 0.:
                slope = new_slope
                break
            elif step < 0. and negative:
                break
            elif step < 0.:
                slope = new_slope
        else:
            print("did not converge")
            
        # Exit codition is met
        
        function = self.segment_factory(slope, logxmin, logxmax, val)
        return function, slope

    def __call__(self, x):
        # left or right boundary inclusive should depend on the creation scheme (going left or right),
        # should set private attribute accordingly that is used in digitize
        idx = np.digitize(x, self.bins) - 1
        func = self._segmented_functions[idx]
        val = func(x)
        if val == -1:
            func = self._segmented_functions[idx - 1]
            val = func(x)
        return val
    
    @abstractmethod
    def generate_segments(self):
        return


class FromPeakSegmentation(SegmentedApprox):
    def __init__(self, target, support):
        # How to?
        bins = 
        super().__init__(target, support, bins)

        # Define resolution order in which the segments are generated
        # start with
```

```python
segment = LinearSegmentedApprox(f, E, breaks)
```

```python
segment.bins, f[0], segment.target_log_approx(1e2), segment.target_log_approx(1e3)
```

```python
segment._segmented_functions[0] = segment._fit_segment(segment.bins[0], segment.bins[1], val=f[0]*1.5)[0]
```

```python
segment(1e3)
```

```python
c = 1
val = segment(segment.bins[c])
print(val)
segment._segmented_functions[c] = segment._fit_segment(segment.bins[c], segment.bins[c+1], val=val)[0]

```

```python
c = 2
val = segment(segment.bins[c])
print(val)
segment._segmented_functions[c] = segment._fit_segment(segment.bins[c], segment.bins[c+1], val=val)[0]
```

```python
c = 3
val = segment(segment.bins[c])
print(val)
segment._segmented_functions[c] = segment._fit_segment(segment.bins[c], segment.bins[c+1], val=val)[0]
```

```python
c = 4
val = segment(segment.bins[c])
print(val)
segment._segmented_functions[c] = segment._fit_segment(segment.bins[c], segment.bins[c+1], val=val)[0]
```

```python
c = 5
val = segment(segment.bins[c])
print(val)
segment._segmented_functions[c] = segment._fit_segment(segment.bins[c], segment.bins[c+1], val=val)[0]
```

```python
segment(1e6)
```

```python
values = [segment(_) for _ in E[:-1]]
plt.plot(E[:-1], values)
plt.plot(E, f)
plt.xscale("log")
plt.yscale("log")
```

```python
Emin = 1e2
Emax = 1e8
E = np.geomspace(Emin, Emax, 1_000)
N = 5

# Define width in decadic unit, half an order of magnitude
width = 1
# get target function
f = sim._exposure_integral[IC86_II]._f_values[0].to_value(u.m**2)
logf = np.log10(f)
logx = np.log10(E)
g = sim._exposure_integral[IC86_II]._g_values[0]
# Find position of max of target function
target_max = f.max()
target_max_point = E[np.argmax(f).squeeze()]
# create break points with max of target as bin center somewhere
middle = np.log10(target_max_point)
breaks = [middle - width / 2, middle + width / 2]
print(breaks)
indices = [0]

def target_log_approx(x):
    return np.power(10, np.interp(np.log10(x), logx, logf))

def segment_factory_high(slope, xmin, xmax, val_at_min):
    # takes boundaries in linear space
    xmin = np.power(10, xmin)
    xmax = np.power(10, xmax)
    print("segment factory", xmin, xmax)
    def func(x):
        return np.power(x / xmin, slope) * val_at_min
    return func

def segment_factory_low(slope, xmin, xmax, val_at_max):
    # takes boundaries in log space
    xmin = np.power(10, xmin)
    xmax = np.power(10, xmax)
    def func(x):
        return np.power(x / xmax, slope) * val_at_max
    return func

def residuals(f, evals):
    return np.power(f - evals, 2)


def init_slope(xmin, xmax):
    return np.log10(target_log_approx(xmax) / target_log_approx(xmin)) / (np.log10(xmax/xmin))




functions = [segment_factory_high(indices[0], *breaks, target_max)]
while True:
    proposal = breaks[-1] + width
    # print(proposal)
    if proposal > np.log10(Emax):
        proposal = np.log10(Emax)
    breaks.append(proposal)

    assert breaks[-1] == proposal

    next_supp = E[np.nonzero((logx <= breaks[-1]) & (logx >=breaks[-2]))]
    low, high = np.power(10, breaks[-2]), np.power(10, breaks[-1])
    prev_seg_value = functions[-1](low)
    print(prev_seg_value)
    print(breaks[-2], breaks[-1])

    proposed_slope = init_slope(low, high)
    print(proposed_slope)

    proposed_func = segment_factory_high(proposed_slope, breaks[-2], breaks[-1], prev_seg_value)
    # functions.append(proposed_func)

    support = np.geomspace(low, high)
    
    diff = proposed_func(support) - target_log_approx(support)

    # if the proposed function under-approximates the target, take a shallower slope
    if np.any(diff < 0.):
        print("under")
        step = 0.05
    else:
        print("over")
        step = -0.05

    trial_functions = []

    while True:
        newly_proposed_slope = proposed_slope + step
        newly_proposed_function = segment_factory_high(newly_proposed_slope, breaks[-2], breaks[-1], prev_seg_value)
        trial_functions.append(newly_proposed_function)
        diff = newly_proposed_function(support) - target_log_approx(support)
        print(np.any(diff < 0.))
        if step > 0 and np.any(diff < 0.):
            # if we are going for shallower slopes and still have negative differences, keep going
            proposed_slope = newly_proposed_slope
            continue
        elif step > 0:
            # going for shallower slopes and there are no negative values, keep the current ones
            proposed_slope = newly_proposed_slope
        elif step < 0 and np.any(diff < 0.):
            # need steeper slope but we have gone too far, previous slope (proposed_slope) was best value already
            pass
        elif step < 0:
            # keep going
            proposed_slope = newly_proposed_slope
            continue

        # exit condition

        functions.append(segment_factory_high(proposed_slope, breaks[-2], breaks[-1], prev_seg_value))
        break

    # break

    # slope -= step
    # break


    if proposal == np.log10(Emax):
        break

    

while True:
    proposal = breaks[0] - width
    if proposal < np.log10(Emin):
        proposal = np.log10(Emin)
    breaks.insert(0, proposal)
    if proposal == np.log10(Emin):
        break
breaks = np.power(10, breaks)

# print(breaks)




plt.plot(E, f)
plt.vlines(breaks, *plt.ylim(), color="black")
#plt.plot(E, functions[0](E))

for c, func in enumerate(functions):
    plt.plot(E, func(E), color=f"C{c}")

plt.vlines(target_max_point, *plt.ylim(), color="red")

plt.xscale("log")
plt.yscale("log")

#plt.xlim(1e5, 1e7)
#plt.ylim(1e-8, 1e-3)
plt.ylim(top = target_max*2)
plt.xlim(1e5, 1e8)
```

```python
Emin = 1e2
Emax = 1e8
E = np.geomspace(Emin, Emax, 1_000)
N = 5

# Define width in decadic unit, half an order of magnitude
width = 1
# get target function
f = sim._exposure_integral[IC86_II]._f_values[0].to_value(u.m**2)
logf = np.log10(f)
logx = np.log10(E)
g = sim._exposure_integral[IC86_II]._g_values[0]
# Find position of max of target function
target_max = f.max()
target_max_point = E[np.argmax(f).squeeze()]
# create break points with max of target as bin center somewhere
middle = np.log10(target_max_point)
breaks = [middle - width / 2, middle + width / 2]
print(breaks)
indices = [0]

def target_log_approx(x):
    return np.power(10, np.interp(np.log10(x), logx, logf))

def segment_factory_high(slope, xmin, xmax, val_at_min):
    # takes boundaries in linear space
    xmin = np.power(10, xmin)
    xmax = np.power(10, xmax)
    print("segment factory", xmin, xmax)
    def func(x):
        return np.power(x / xmin, slope) * val_at_min
    return func

def segment_factory_low(slope, xmin, xmax, val_at_max):
    # takes boundaries in log space
    xmin = np.power(10, xmin)
    xmax = np.power(10, xmax)
    def func(x):
        return np.power(x / xmax, slope) * val_at_max
    return func

def residuals(f, evals):
    return np.power(f - evals, 2)


def init_slope(xmin, xmax):
    return np.log10(target_log_approx(xmax) / target_log_approx(xmin)) / (np.log10(xmax/xmin))




functions = [segment_factory_high(indices[0], *breaks, target_max)]
while True:
    proposal = breaks[-1] + width
    # print(proposal)
    if proposal > np.log10(Emax):
        proposal = np.log10(Emax)
    breaks.append(proposal)

    assert breaks[-1] == proposal

    next_supp = E[np.nonzero((logx <= breaks[-1]) & (logx >=breaks[-2]))]
    low, high = np.power(10, breaks[-2]), np.power(10, breaks[-1])
    prev_seg_value = functions[-1](low)
    print(prev_seg_value)
    print(breaks[-2], breaks[-1])

    proposed_slope = init_slope(low, high)
    print(proposed_slope)

    proposed_func = segment_factory_high(proposed_slope, breaks[-2], breaks[-1], prev_seg_value)
    # functions.append(proposed_func)

    support = np.geomspace(low, high)
    
    diff = proposed_func(support) - target_log_approx(support)

    # if the proposed function under-approximates the target, take a shallower slope
    if np.any(diff < 0.):
        print("under")
        step = 0.05
    else:
        print("over")
        step = -0.05

    trial_functions = []

    while True:
        newly_proposed_slope = proposed_slope + step
        newly_proposed_function = segment_factory_high(newly_proposed_slope, breaks[-2], breaks[-1], prev_seg_value)
        trial_functions.append(newly_proposed_function)
        diff = newly_proposed_function(support) - target_log_approx(support)
        print(np.any(diff < 0.))
        if step > 0 and np.any(diff < 0.):
            # if we are going for shallower slopes and still have negative differences, keep going
            proposed_slope = newly_proposed_slope
            continue
        elif step > 0:
            # going for shallower slopes and there are no negative values, keep the current ones
            proposed_slope = newly_proposed_slope
        elif step < 0 and np.any(diff < 0.):
            # need steeper slope but we have gone too far, previous slope (proposed_slope) was best value already
            pass
        elif step < 0:
            # keep going
            proposed_slope = newly_proposed_slope
            continue

        # exit condition

        functions.append(segment_factory_high(proposed_slope, breaks[-2], breaks[-1], prev_seg_value))
        break

    # break

    # slope -= step
    # break


    if proposal == np.log10(Emax):
        break

    

while True:
    proposal = breaks[0] - width
    if proposal < np.log10(Emin):
        proposal = np.log10(Emin)
    breaks.insert(0, proposal)
    if proposal == np.log10(Emin):
        break
breaks = np.power(10, breaks)

# print(breaks)




plt.plot(E, f)
plt.vlines(breaks, *plt.ylim(), color="black")
#plt.plot(E, functions[0](E))

for c, func in enumerate(functions):
    plt.plot(E, func(E), color=f"C{c}")

plt.vlines(target_max_point, *plt.ylim(), color="red")

plt.xscale("log")
plt.yscale("log")

#plt.xlim(1e5, 1e7)
#plt.ylim(1e-8, 1e-3)
plt.ylim(top = target_max*2)
plt.xlim(1e5, 1e8)
```

```python
functions
```

```python
init_slope(1e5, 1e8)
```

```python
"""E = np.geomspace(1e2, 1e8)
f = sim._exposure_integral[IC86_II]._f_values[0]
g = sim._exposure_integral[IC86_II]._g_values[0]
plt.plot(E, f)
plt.plot(E, g)

plt.plot(E, f / g)

plt.xscale("log")
plt.yscale("log")
"""
```

```python
my_sources.point_source[0].parameters
```

```python
sim._Nex_et
```

```python
sim._Nex_et.tolist()
```

```python
sim.generate_stan_code()
```

```python
sim.compile_stan_code()
```

```python
inputs = sim._get_sim_inputs()
inputs["rs_bbpl_Eth"], inputs["rs_bbpl_gamma1"], inputs["rs_cvals"]
```

```python
inputs = sim._get_sim_inputs()
inputs["rs_bbpl_Eth"], inputs["rs_bbpl_gamma1"], inputs["rs_cvals"]
```

([50000.0],
 [-0.8],
 [[1.1001377229425167, 0.6348285313354258, 0.008227014924210274]])

```python
sim.run(verbose=True, show_console=True)
```

```python
sim._sim_output.stan_variable("event")
```

```python
sim.show_skymap()
sim.show_spectrum()
```

For each physical parameter there is a seperate prior class implemented, e.g. for the luminosity a `LuminosityPrior`. We can pass different types of distributions, namely `NormalPrior`, `LogNormalPrior` and for the luminosity only `ParetoPrior` with appropriate $\mu, \sigma$ (or $x_{min}, \alpha$). The priors convert the passed units to the internally used ones. `IndexPriors` do not use units. Fluxes are integrated number fluxes, i.e. dimension is 1 / area / time.

```python
priors = Priors()
priors.diffuse_flux = FluxPrior(LogNormalPrior, mu=my_sources.diffuse.flux_model.total_flux_int, sigma=0.5)
priors.luminosity = LuminosityPrior(
    mu=L.value / sim._expected_Nnu_per_comp[0] * 10,   # we expect ~10 events
    sigma=2
)
priors.diff_index = IndexPrior(mu=2.52, sigma=0.08)
priors.src_index = IndexPrior(mu=2.5, sigma=1)
```

```python
# Copy flux
atmo_flux_int = my_sources.atmospheric.flux_model.total_flux_int
Nex = np.sum(sim._expected_Nnu_per_comp[2])
Nex_std = np.sqrt(Nex)
N = events.N
N_std = np.sqrt(N)
# Nex is mean of poisson and at zeroth order all background
# std is sqrt(Nex) -> add to Nex and convert to flux
# this flux is then 1sigma above mean
mu = atmo_flux_int / Nex * N
sigma = atmo_flux_int / Nex * np.sqrt(N)
# N +/- N_std -> N_std is sigma on scale of events
# atmo_flux_int +/- sigma should yield N +/- N_std events
priors.atmospheric_flux = FluxPrior(mu=mu, sigma=sigma)
```

```python
fit = StanFit(my_sources, event_types, events, lifetime, nshards=40)
```

```python
fit.precomputation()
```

```python
fit.generate_stan_code()
fit.compile_stan_code()
```

Run the fit with some appropriate initial values. With 40 threads, this will take roughly 6 minutes (subject to change, hopefully for the better).

```python
fit.run(
    show_progress=True, inits={"L": 1e48, "src_index": 2.2, "diff_index": 2.2, "F_atmo": 1e-1, "F_diff": 1e-5}, show_console=True,
)
```

```python
fit.save("txs_pgamma.h5", overwrite=True)
```

```python
fit.plot_flux_band(upper_limit=True)
```

```python
fit.save("txs_pgamma.h5", True)
```

```python
fit = StanFit.from_file("txs_pgamma.h5")
```

```python
fit.plot_trace()
```

```python
fit.plot_trace_and_priors()
```

```python
fit.plot_trace_and_priors("E0_src", transform=True)
```

```python
fit.plot_energy_and_roi()
```

```python
config = HierarchicalNuConfig.make_config(my_sources)
```

```python
import h5py
```

```python
with h5py.File("txs_pgamma.h5", "r") as f:
    print(f.keys())
```

```python
fit = StanFit.from_file("txs_pgamma.h5")
fit.plot_flux_band(upper_limit=True)
```

```python

```

```python
import h5py
```

```python
with h5py.File("txs_pgamma.h5", "r") as f:
    print(f["sources/config"][()].decode("ascii"))
```

```python
fit.plot_trace_and_priors(fit._def_var_names+["E_peak", "Nex_src"])
fig, axs = fit.plot_energy_and_roi()
# fig.savefig("txs_flare_pl_roi.pdf")
```

```python
fig, axs = fit.plot_trace_and_priors(["E0_src", "E_peak", "peak_energy_flux"], transform=lambda x: np.log10(x))
```

```python
fit = StanFit.from_file("fit_config_1721398047.h5")
```

```python
Epeak = fit._fit_output["E_peak"].squeeze()
peak_flux = fit._fit_output["peak_energy_flux"].squeeze()
mask = peak_flux > 0.
data = {"energy": np.log10(Epeak[mask]), "flux": np.log10(peak_flux[mask])}
res = sns.kdeplot(
    data=data,
    x="energy",
    y="flux",
    fill=False,
    alpha=1.,
    #ax=ax,
    cmap="Wistia",
    # cbar=True,
    # cbar_kws={"label": "kde", },
    levels=[0.05, 0.217, 0.5, 1.],
    thresh=1e-3,
)

contours = [_[0] for _ in res.collections[0].allsegs]
```

```python
colors = res.collections[0]._mapped_colors
confidence = 1 - np.array([0.05, 0.217, 0.5, 1.])[:-1]
```

```python
fig, ax = fit.plot_flux_band(2., [0.5, 0.683, 0.95,], energy_unit=u.GeV, area_unit=u.m**2)

for cont, c, ci in zip(contours, colors, confidence):
    try:
        ax.plot(np.power(10, cont[:, 0]), np.power(10, cont[:, 1]), color=c, label=f"{ci}\% CI")
    except:
        continue
#ax.set_xlim(2e3, 3e6)
#ax.set_ylim(1e-6, 1e-2);
ax.legend()
```

```python
res.collections[0].allsegs
```

```python
ax = sns.kdeplot(data=data, x="energy", y="flux", fill=True, cmap="viridis_r", alpha=0.5)
```

```python

```

```python
ic_events = np.loadtxt("ic_txs_flare_events.dat", dtype=str)
```

```python
ic_events[(0, 1, 2, 3),]
```

```python
cols = (0, 5, 6, 7, 8)
rows = (0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16)
ic = ic_events[rows,][:, cols].astype(float)
```

```python
np.argwhere(np.isclose(events.mjd.value, ic[0, 0], atol=1) & np.isclose(np.log10(events.energies.to_value(u.GeV)), ic[0, -1], atol=0.01))
```

```python
events.energies[26], np.power(10, ic[0, -1])
```

```python
fit.plot_flux_band()
```

```python
fig, ax = fit.plot_flux_band(2.0, 0.68, u.erg, u.cm**2, u.GeV)
ax.set_xlim(1e3, 1e7)
ax.set_ylim(1e-14, 1e-10)
```

```python
import h5py
```

```python
with h5py.File("txs_flare_pl.h5") as f:
    print(f.keys())
    print(f["sources/config"][()].decode("ascii"))
```

```python
fit.save("txs_flare_pl.h5", overwrite=True)
```

```python
fit_pl = StanFit.from_file("txs_flare_pl.h5")
```

```python
fit_pl.plot_flux_band(2., upper_limit=False)
```

```python
fit_pl.get_src_position(0)
```

```python
fit_pl.plot_energy_and_roi()
```

```python
fit_pl.events.N
```

```python
import arviz as av
import matplotlib.pyplot as plt
```

```python
ref_nex = np.array([10.00, 4.2, 5.2])
ref_index = np.array([2.2, 0.3, 0.3])   # https://arxiv.org/pdf/2109.05818.pdf as reference, but they use a sliding gaussian to actually determine the time window, so be careful
ref = {"src_index": ref_index, "Nex_src": ref_nex}
```

```python
keys = ["L", "src_index", "Nex_src"]
label = [r"$L~[\si{\erg\per\second}]$", r"$\gamma$", "Nex"]
transformations = [lambda x: x * (u.GeV / u.erg).to(1), lambda x: x, lambda x: x]
CL = [0.5, 0.683]

hdi = {key: [av.hdi(transformations[c](fit_pl._fit_output[key].T), hdi_prob=_).flatten() for _ in CL] for c, key in enumerate(keys)}
kdes = {key: av.kde(transformations[c](fit_pl._fit_output[key])) for c, key in enumerate(keys)}
```

```python
fig, axs = plt.subplots(nrows=len(keys), ncols=1, figsize=(2.80278, 4), gridspec_kw={"hspace": .80})

for c, key in enumerate(keys):
    ax = axs[c]
    kde = kdes[key]
    HDI = hdi[key]
    ax.plot(*kde)
    for HDI in hdi[key]:
        start = np.digitize(HDI[0], kde[0]) - 1
        stop = np.digitize(HDI[1], kde[0])
        ax.fill_between(kde[0][start:stop], np.zeros_like(kde[0][start:stop]), kde[1][start:stop], color="C0", alpha=0.3)
    try:
        errs = ref[key]
        ax.errorbar(errs[0], np.sum(ax.get_ylim())*0.45, xerr=errs[1:3][:, np.newaxis], color="black", capsize=4, fmt="x")
    except KeyError:
        pass
    ax.set_yticks([])
    ax.set_xlabel(label[c])
```

```python
fig.savefig("txs_flare_pl_posteriors.pdf")
```

```python
import h5py
```

```python
with h5py.File("/home/iwsatlas1/kuhlmann/icecube/hierarchical_nu/examples/txs_with_config.h5", "r") as f:
    config = OmegaConf.create(f["sources/config"][()].decode("ascii"))
```

```python
parser = ConfigParser(config)
```

```python
sources = parser.sources
```

```python
sources.point_source[0].ra.to(u.deg)
```

```python
fit = StanFit.from_file("txs_with_config.h5")
```

```python
fit.plot_flux_band()
```

```python
fit._power_law
```

Confirm that there are no problems encountered by the HMC.

```python
fit.diagnose()
```

```python
fit.save("txs_pgamma.h5")
```

Display the posterior distributions alongside the inputted prior distributions, where available.

Compare the results to those of a likelihood analysis by IceCube: ns=11.87, gamma=2.22

```python
fit.plot_trace_and_priors(var_names=fit._def_var_names+["Nex_src", "Nex_diff", "Nex_atmo"])
```

```python
fit.save("txs_powerlaw.h5")
```

```python
fit = StanFit.from_file("txs_pgamma.h5")
```

Marginalising over all posteriors, the association probability of each event with a source component can be extracted. Displayed in the following plots are a spatial and an energetic overview of all events. The blob size is arbitrary. The colorscale encodes the association probability, here to the proposed point source marked by a grey cross in the left plot.

On the right, the energy posteriors of each event is shown. The distributions are transformed to $\log_{10}(E / \text{GeV})$. At the top, reconstructed energies of all selected events are shown as short vertical lines. Above an association probability of 20% they are linked by a dashed line to the corresponding posterior distribution.

```python
fit.plot_energy_and_roi()
```

```python
#fit._sources = my_sources
#fit._logparabola = False
#fit._pgamma = True
fig, ax = fit.plot_flux_band(2, [0.5, 0.95])
ax.set_xlim(1e3, 1e6)
ax.set_ylim(1e-14, 1e-10)
```

```python

```

```python
fit.plot_trace_and_priors("E0_src", transform=lambda x: np.log10(x))
```

Lastly, show the correlations between the parameters.

```python
fit.corner_plot()
```

```python

```
