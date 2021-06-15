---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: hierarchical_nu
    language: python
    name: hierarchical_nu
---

## Generate files for detector model from simulation files

Want to have combined Aeff and Eres for all channels and flavours to keep things simple. Also will add in HESE cut manually after for consistency with tracks and easy editing. 

```python
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
```

#### Aeff 

```python
e_bins = np.linspace(2, 8.5, 51)
cosz_bins = np.linspace(-1, 1, 11)

path = "simulation_files"
filenames = os.listdir(path)
```

```python
# Total Aeff
aeff_list = []
for name in filenames:
    
    # Ignore track-like signatures 
    if not "numu_CC" in name and not "numubar_CC" in name:
    
        # Get histogram
        df = pd.read_hdf(os.path.join(path, name))
        w = df['generation_weight'] * 1e-4 / (2.*np.pi)
        h, xbins, ybins = np.histogram2d(np.log10(df["prim_energy"]), 
                                         df["prim_coszenith"], 
                                        bins=[e_bins, cosz_bins], weights=w)
        # Normalise to bin area
        aeff = np.zeros_like(h)
        for i, l10E in enumerate(xbins[:-1]):
            for j, cosz in enumerate(ybins[:-1]):
                bin_area = ((10**xbins[i+1] - 10**l10E) * (ybins[j+1] - cosz))
                aeff[i][j] = h[i][j] / bin_area
                
        aeff_list.append(aeff)
        
aeff_tot = np.sum(aeff_list, axis=0) / 6.0 # flavours and nu/nubar
```

```python
fig, ax = plt.subplots()
cf = ax.contourf(e_bins[:-1], cosz_bins[:-1], aeff_tot.T, levels=20)
cbar = fig.colorbar(cf)
ax.set_xlabel("log10(Etrue [GeV])")
ax.set_ylabel("cosz")
#ax.set_xlim(4.5, 7.0)
cbar.set_label("Effective Area [m^2]", labelpad=10)
```

```python
# To weight Eres contributions
aeff_sum_list = [np.sum(aeff) for aeff in aeff_list]
aeff_sum_list = aeff_sum_list / max(aeff_sum_list)
```

#### Eres

```python
tE_bins = np.linspace(np.log10(3e4), 7, 51)
rE_bins = np.linspace(3, 7, 51)
```

```python
# Total Eres 
eres_list = []
for name in filenames:
    
    # Ignore track-like signatures 
    if not "numu_CC" in name and not "numubar_CC" in name:
        
        # Get histogram
        df = pd.read_hdf(os.path.join(path, name))
        h, xbins, ybins = np.histogram2d(np.log10(df['prim_energy']),
                                         np.log10(df['rec_energy']), 
                                         bins=[tE_bins, rE_bins])
        eres_list.append(h)
        
eres_list = [eres * w for eres, w in zip(eres_list, aeff_sum_list)]
eres_tot = np.sum(eres_list, axis=0) 

# Norm along Ereco
bin_width = ybins[1] - ybins[0]
for i, pdf in enumerate(eres_tot):
    eres_tot[i] = pdf / (pdf.sum() * bin_width)
```

```python
fig, ax = plt.subplots()
cf = ax.contourf(xbins[:-1], ybins[:-1], eres_tot.T, levels=20)
ax.set_xlabel("log10(Etrue [GeV])")
ax.set_ylabel("log10(Ereco [GeV])")
cbar = fig.colorbar(cf)
cbar.set_label("P(Ereco|Etrue)")
```

```python
fig, ax = plt.subplots()
ax.plot(rE_bins[:-1], eres_tot[30])
```

#### Save to file

```python
import h5py
with h5py.File("cascade_detector_model.h5", "w") as f:
    
    aeff_folder = f.create_group("aeff")
    aeff_folder.create_dataset("tE_bin_edges", data=10**e_bins)
    aeff_folder.create_dataset("cosz_bin_edges", data=cosz_bins)
    aeff_folder.create_dataset("aeff", data=aeff_tot)
    
    eres_folder = f.create_group("eres")
    eres_folder.create_dataset("tE_bin_edges", data=10**tE_bins)
    eres_folder.create_dataset("rE_bin_edges", data=10**rE_bins)
    eres_folder.create_dataset("eres", data=eres_tot)
```

## Check expected atmospheric contribution

Deciding whether to include an atmospheric component for cascade events above Ereco of 60 TeV. Check by making a detector model file with only numu_NC and numubar_NC contributions.

```python
from astropy import units as u
from hierarchical_nu.source.parameter import Parameter
from hierarchical_nu.source.source import Sources
```

```python
e_bins = np.linspace(np.log10(3e4), 8.5, 51)
cosz_bins = np.linspace(-1, 1, 11)

path = "simulation_files"
filenames = os.listdir(path)
```

```python
# Total Aeff
aeff_mu = []
for name in filenames:
    
    # Ignore everything else 
    if "numu_NC" in name or "numubar_NC" in name:
    
        # Get histogram
        df = pd.read_hdf(os.path.join(path, name))
        w = df['generation_weight'] * 1e-4 / (2.*np.pi)
        h, xbins, ybins = np.histogram2d(np.log10(df["prim_energy"]), 
                                         df["prim_coszenith"], 
                                        bins=[e_bins, cosz_bins], weights=w)
        # Normalise to bin area
        aeff = np.zeros_like(h)
        for i, l10E in enumerate(xbins[:-1]):
            for j, cosz in enumerate(ybins[:-1]):
                bin_area = ((10**xbins[i+1] - 10**l10E) * (ybins[j+1] - cosz))
                aeff[i][j] = h[i][j] / bin_area
                
        aeff_mu.append(aeff)
        

        aeff_mu_tot = np.sum(aeff_mu, axis=0) / 2 # nu/nubar
```

```python
fig, ax = plt.subplots()
cf = ax.contourf(e_bins[:-1], cosz_bins[:-1], aeff_mu_tot.T, levels=20)
cbar = fig.colorbar(cf)
ax.set_xlabel("log10(Etrue [GeV])")
ax.set_ylabel("cosz")
cbar.set_label("Effective Area [m^2]", labelpad=10)
```

```python
# To weight Eres contributions
aeff_sum_list_mu = [np.sum(aeff) for aeff in aeff_mu]
aeff_sum_list_mu = aeff_sum_list_mu / max(aeff_sum_list_mu)
```

```python
tE_bins = np.linspace(np.log10(3e4), 7, 51)
rE_bins = np.linspace(3, 7, 51)
```

```python
# Total Eres 
eres_mu = []
for name in filenames:
    
    # Ignore not mu_NC 
    if "numu_NC" in name or "numubar_NC" in name:
        
        # Get histogram
        df = pd.read_hdf(os.path.join(path, name))
        h, xbins, ybins = np.histogram2d(np.log10(df['prim_energy']),
                                         np.log10(df['rec_energy']), 
                                         bins=[tE_bins, rE_bins])
        eres_mu.append(h)
        
eres_mu = [eres * w for eres, w in zip(eres_mu, aeff_sum_list_mu)]
eres_mu_tot = np.sum(eres_mu, axis=0) 

# Norm along Ereco
bin_width = ybins[1] - ybins[0]
for i, pdf in enumerate(eres_mu_tot):
    eres_mu_tot[i] = pdf / (pdf.sum() * bin_width)
```

```python
fig, ax = plt.subplots()
cf = ax.contourf(xbins[:-1], ybins[:-1], eres_mu_tot.T, levels=20)
ax.set_xlabel("log10(Etrue [GeV])")
ax.set_ylabel("log10(Ereco [GeV])")
cbar = fig.colorbar(cf)
cbar.set_label("P(Ereco|Etrue)")
```

```python
fig, ax = plt.subplots()
ax.plot(rE_bins[:-1], eres_mu_tot[30])
```

#### Save to file

```python
import h5py
with h5py.File("cascade_detector_model_test_mu_NC.h5", "w") as f:
    
    aeff_folder = f.create_group("aeff")
    aeff_folder.create_dataset("tE_bin_edges", data=10**e_bins)
    aeff_folder.create_dataset("cosz_bin_edges", data=cosz_bins)
    aeff_folder.create_dataset("aeff", data=aeff_mu_tot)
    
    eres_folder = f.create_group("eres")
    eres_folder.create_dataset("tE_bin_edges", data=10**tE_bins)
    eres_folder.create_dataset("rE_bin_edges", data=10**rE_bins)
    eres_folder.create_dataset("eres", data=eres_mu_tot)
```

```python

```
