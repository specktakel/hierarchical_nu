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

```python
import sys
sys.path.append("../../")
```

```python
import numpy as np
from matplotlib import pyplot as plt

from hierarchical_nu.model_check import ModelCheck
```

```python
model_check = ModelCheck()
```

```python
file_stem = "output/"
file_list = [file_stem+"fit_sim_1234_test.h5"]

model_check.load(file_list)
```

```python
fig, ax = model_check.compare(show_prior=True)
```

```python

```
