## Simulation study

In order to verify the implementation of the statistical model, we can fit simulations and check that it is possible to recover the known truth. The `ModelCheck` class in hierarchical_nu is an easy interface for running sets of simulations and fits for different parameters. 

### The configuration files

The configuration of the simulation study is handled via the `FileConfig` and `ParameterConfig` classes in the `hierarchical_nu.utils.config` module. Running `initialise.py 1` will generate the default YAML config files, as specified in the `config` module. These will be placed in `~/.hierarchical_nu` and can be editied later to change the input parameter values. See the comments in `hierarchical_nu.utils.config` for units and explanations.

Make sure that your choice of parameters makes sense and doesn't lead to a ridiculous number (>1000) of neutrino events which could result in very slow simulations and fits. Ideally, first check the choice is sensible by running a notebook similar to the `simulate_and_fit` example. 

### How to run

Once you have set up the parameters for your simulation in the configuration file, first run the initialisation script. This will make sure that all heavy computations are run and cached locally, output folders are created, and Stan code is precompiled. Configuration files will also be generated if they do not exist.

```
python initialise.py n_nodes
```

Here, the `n_nodes` argument is used to define the number of seeds needed for different nodes, if running on a large cluster with SLURM. If not, just enter 1.

If running locally, or on the ODSL server, launch separate processes with

```
python simulation_study.py n_jobs n_subjobs seed
```

Where `n_jobs` is the number of parallel processes, and n_subjobs is the number of simulation/fit runs in serial on each process. Also provide a seed, the code will take care that each run has different seeds based on your starting seed. You can edit the `simulation_study.py` script to sepcify the output filename. The total number of independent simulation/fit runs will be `n_jobs` x `n_subjobs`. 

If running on a cluster with SLURM, use the `parallel_run.sh` job script. Edit the script to set the number of nodes, jobs per node and subjobs per process.

```
sbatch parallel_run.sh
```

The total number of independent simulation/fit runs will be `n_node` x `n_jobs` x `n_subjobs`.

## Check results

See the `check_results` notebook for how to use the `ModelCheck` class to have a look at the results.
