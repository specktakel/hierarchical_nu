---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: bayes
    language: python
    name: bayes
---

```python
import numpy as np
from matplotlib import pyplot as plt
import math
import sys
sys.path.append("../../")

from hierarchical_nu.model_check import ModelCheck
```

```python
model_check = ModelCheck()
```

```python
file_stem = "output/icecube_tests/"
file_list = [file_stem+"fit_sim_icecube_1000_singlesource_newprior.h5",
             file_stem+"fit_sim_icecube_9900_singlesource_newprior.h5"]
#file_list = [file_stem+"fit_sim_42_test.h5"]

model_check.load(file_list)
```

```python
fig, ax = model_check.compare(show_prior=True)
```

```python
#plt.style.use("minimalist")
```

```python
#fig.savefig("figures/catalog_sim_study.pdf", dpi=500, bbox_inches="tight")
```

## Compare different detector models

```python
import h5py
```

```python
with h5py.File("../output/icecube_test.h5", "r") as f:
    f_ic = f["fit/outputs/f"][()]
    alpha_ic = f["fit/outputs/alpha"][()] 
    
with h5py.File("../output/cascades_test.h5", "r") as f:
    f_c = f["fit/outputs/f"][()]
    alpha_c = f["fit/outputs/alpha"][()] 
    
with h5py.File("../output/northern_tracks_test.h5", "r") as f:
    f_t = f["fit/outputs/f"][()]
    alpha_t = f["fit/outputs/alpha"][()]    
```

```python
plt.style.use("minimalist")
```

```python
# Associated fraction
bins = np.linspace(0, 0.2)
fig, ax = plt.subplots()
ax.hist(f_t, bins=bins, alpha=0.5, label="Northern tracks", color="#017B76", density=True)
ax.hist(f_c, bins=bins, alpha=0.5, label="Cascades", color="#99195E", density=True)
ax.hist(f_ic, bins=bins, alpha=0.5, label="Joint", color="k", density=True)
ax.legend()
ax.axvline(0.04172, color='k')
ax.set_xlabel("$f$")
fig.savefig("figures/compare_dm_f.pdf", dpi=500, bbox_inches='tight')
```

```python
# Spectral index
bins = np.linspace(1.25, 2.75)
fig, ax = plt.subplots()
ax.hist(alpha_t, bins=bins, alpha=0.5, label="Northern tracks", color="#017B76", density=True)
ax.hist(alpha_c, bins=bins, alpha=0.5, label="Cascades", color="#99195E", density=True);
ax.hist(alpha_ic, bins=bins, alpha=0.5, label="Joint", color="k", density=True)
ax.legend()
ax.axvline(2.0, color='k')
ax.set_xlabel("$\\alpha$")
fig.savefig("figures/compare_dm_alpha.pdf", dpi=500, bbox_inches='tight')
```

```python

```
