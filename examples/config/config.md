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

## Demonstration of using a config file for simulation and fit

```python
from hierarchical_nu.utils.config_parser import ConfigParser
from hierarchical_nu.utils.config import HierarchicalNuConfig
from hierarchical_nu.source.source import uv_to_icrs
from hierarchical_nu.events import Events
from pathlib import Path
from hierarchical_nu.detector.icecube import IC86_II
from hierarchical_nu.utils.roi import ROIList
import numpy as np
import astropy.units as u

```

```python
config_path = Path("./hnu_config.yml")
```

```python
config = HierarchicalNuConfig.load_default()
```

```python
config
```

```python
parser = ConfigParser(config)
```

```python
sources = parser.sources
```

```python
dm = parser.detector_model
obs_time = parser.obs_time
```

```python
dm
```

```python
obs_time = parser.obs_time
```

```python
obs_time
```

```python
parser.ROI
```

```python
sim = parser.create_simulation(sources, dm, obs_time)
```

```python
sim.precomputation()
sim.generate_stan_code()
sim.compile_stan_code()
# sim.setup_stan_sim(".stan_files/sim_code")
```

```python
sim._stan_interface._ps_frame
```

```python
sim._get_expected_Nnu(sim._get_sim_inputs())
```

```python
sim._Nex_et
```

```python
sim.run()
```

```python
sim.show_skymap()
sim.show_spectrum()
```

```python
sim.save("sim.h5", overwrite=True)
```

```python
events = Events.from_file("sim.h5")
```

```python
events.N
```

```python
fit = parser.create_fit(sources, events, dm, obs_time)
```

```python
fit.precomputation()
fit.generate_stan_code()
fit.compile_stan_code()
# fit.setup_stan_fit(".stan_files/model_code")
```

```python
fit.run(
    show_progress=True,
    show_console=True,
    inits={
        "L": 1e49,
        "src_index": 2.2,
        "beta_index": 0.05,
        "E0_src": 1e5,
        "diff_index": 2.2,
        "E": [1e5] * fit.events.N,
        "F_diff": 1e-4,
        "F_atmo": 0.3
    }
)
```

```python
fit.plot_energy_and_roi()
```

```python
fit.plot_trace_and_priors()
```

```python

```
