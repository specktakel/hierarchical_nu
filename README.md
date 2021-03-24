# hierarchical_nu

A Bayesian hierarchical model for source-nu associations.

## Installation

The package can currently be installed with:

```
pip install git+https://github.com/cescalara/hierarchical_nu.git
```

In future, hierachical_nu will also be available on PyPI!

The above command will go ahead and install any dependencies that you may be missing to run the core code. 

### Setting up Stan

The hierarchical model is implemented in [Stan](https://mc-stan.org), using the [CmdStan](https://github.com/stan-dev/cmdstan) and [CmdStanPy](https://github.com/stan-dev/cmdstanpy) interfaces. CmdStanPy will be installed as needed using pip if you follow the above instructions. However if you have not set up and compiled CmdStan before, the extra step detailed below is needed. See the CmdStanPy [installation docs](https://mc-stan.org/cmdstanpy/installation.html) for more information.


You can set up CmdStan by running the following python code:

```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```

Or via the command line on MacOS/Linux:

```
install_cmdstan
```

This will make and install CmdStan in the `~/.cmdstan` directory.

## Examples

You can find some example notebooks stored as markdown files in the `examples/` directory. To run these notebooks, use the [jupytext](https://github.com/mwouts/jupytext) package to open the markdown files.

The first time that you use hierarchical_nu, some longer calculations will be run and cached locally. This is a one-time cost, so please be patient. 

More documentation coming soon!


## Running on the ODSL server


### Local install

Start by installing your favourite python virtualenv interface.

```
pip install --user virtualenvwrapper
```

Make sure you also set any necessary paths in your shell profile. Make a new virtualenv and then follow the above installation steps.


### Running through a singularity container - coming soon!

For easy use, there is a docker container provided in `docker/`. Instructions on how to re-build the docker image and convert to singularity format are also provided there, but there is a public version available [here](https://hub.docker.com/repository/docker/cescalara/hierarchical-nu) on Docker Hub. This can be used on the ODSL server through [singularity](https://sylabs.io/guides/3.7/user-guide/index.html). The image is located at `/remote/ceph2/user/f/fran/containers/hierarchical-nu_latest.sif`. You can make a symlink to access it locally with

```
ln -s /remote/ceph2/user/f/fran/containers/hierarchical-nu_latest.sif local/path
```

You can launch a shell within this image using:

```
singularity shell hierarchical-nu_latest.sif
```

Or simply execute a single command within the container using:

```
singularity exec hierarchical-nu_latest.sif command
```

When inside the container, you have the same username as on the machine, and singularity mounts `/home/$USER`, `/tmp`, and `$PWD` into your container by default. You can add additional directories via the `--bind` command. Please see the [singularity docs](https://sylabs.io/guides/3.7/user-guide/index.html) for more information.

### Remote jupyter notebook

As a "hello world" example, you can try to run the `simulate_and_fit` notebook in `examples`. Clone this repo to the server and run

```
jupyter-notebook --no-browser --port=8080
```

Then, on your local machine run

```
ssh -N -f -L localhost:8888:localhost:8080 user@odslserv01.mpp.mpg.de
```

and navigate to `localhost:8888` in your browser. From here, find the `simulate_and_fit` example to check things are running.
