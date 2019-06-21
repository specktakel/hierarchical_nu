FROM python:3.7-slim

# Basic tools
RUN apt-get update && apt-get install -y git build-essential libgeos-dev

# python
RUN pip install numpy scipy ipython jupyter pandas matplotlib astropy pystan h5py stan_utility vMF corner seaborn
RUN pip install -U git+https://github.com/matplotlib/basemap.git
RUN pip install skymap fancy tqdm

# For stan cache
RUN mkdir /root/.stan_cache

# Run jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--allow-root"]
