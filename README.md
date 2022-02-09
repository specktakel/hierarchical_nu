# hierarchical_nu

![CI](https://github.com/cescalara/hierarchical_nu/actions/workflows/tests.yml/badge.svg)

A Bayesian hierarchical model for source-nu associations.

## Installation

The package can currently be installed with:

```
pip install git+https://github.com/cescalara/hierarchical_nu.git
```

In future, hierachical_nu will also be available on PyPI!

The above command will go ahead and install any dependencies that you may be missing to run the core code. If you want to use the optional popsynth interface to simulate populations, then you can also run:

```
pip install git+https://github.com/cescalara/popsynth.git
```

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

### A note on updating existing code

For a clean install, be aware that some calculations are cached in your local working directory when your run the code. Please delete any files in `.cache/` and the necessary calculations will be re-run as you go along.

## Examples

You can find some example notebooks stored as markdown files in the `examples/` directory. To run these notebooks, use the [jupytext](https://github.com/mwouts/jupytext) package to open the markdown files.

The first time that you use hierarchical_nu, some longer calculations will be run and cached locally. This is a one-time cost, so please be patient. 

More documentation coming soon!


## Running on the ODSL server

Once you have an MPI account set up, access through 

```
ssh <user>@odslserv01.mpp.mpg.de
```

where `<user>` is your username.

### Using the ODSL singularity container

Our sysadmin maintains a singularity container for us to run on. To set it up, follow these steps.
* Add `export PATH="/.hb/raidg01/sw/venv/noarch/current/bin:$PATH"` to your `~/.bashrc`, this will let you access the useful `venv` executable
* Make a folder in your home directory with the name of the venv e.g. `~/.venv/odsl`
* Symlink the singularity image to this directory with `ln -s /remote/ceph2/group/odsl/vm/singularity/images/mppmu_odsl-ml_latest.sif ~/.venv/odsl/rootfs.sif`

You can now launch the container with `venv odsl`. This will mount the container `user` directory to `/user/` and your home directory to `/homedir/`. 

You can now proceed as you would with a local install, as described below.

**NB: There are still some issues with the compilation of Stan in the container, so a simple local install is recommend for now (see below)**

### Local install

Start by installing your favourite python virtualenv interface. For example...

```
pip install --user virtualenvwrapper
```

Make sure you also set any necessary paths in your shell profile. Make a new virtualenv and then follow the above installation steps.

### Remote jupyter notebook

As a "hello world" example, you can try to run the `simulate_and_fit` notebook in `examples`. Clone this repo to the server and run

```
jupyter-notebook --no-browser --port=<remote_port>
```

Where `<remote_port> is something like 8080`. Then, on your local machine run

```
ssh -N -f -L localhost:<local_port>:localhost:<remote_port> <user>@odslserv01.mpp.mpg.de
```

and navigate to `localhost:<local_port>` in your browser. You should have make sure that `<local_port>` is free and. different from `<remote_port>`. From here, find the `simulate_and_fit` example to check things are running.
