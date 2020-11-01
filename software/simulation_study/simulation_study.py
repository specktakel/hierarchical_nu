import numpy as np
import h5py
import time
import sys
from joblib import Parallel, delayed
from astropy import units as u

sys.path.append("../")

from python.source.parameter import Parameter
from python.source.source import Sources, PointSource
from python.simulation import Simulation
from python.detector_model import NorthernTracksDetectorModel
from python.events import Events
from python.fit import StanFit

"""
Script to run fits to simulations for 
a selection of known input parameters.
"""


def run_fit():

    N = 1
    include_paths = ["/Users/fran/projects/hierarchical_nu/software/stan_files"]
    fit_filename = "/Users/fran/projects/hierarchical_nu/software/simulation_study/stan_files/model_code.stan"
    atmo_sim_filename = "/Users/fran/projects/hierarchical_nu/software/simulation_study/stan_files/atmo_gen.stan"
    main_sim_filename = "/Users/fran/projects/hierarchical_nu/software/simulation_study/stan_files/sim_code.stan"

    start_time = time.time()

    # Simulation parameters
    alpha = 2.3
    lumi = 1e47 * u.erg / u.s

    Enorm_val = 1e5 * u.GeV
    Emin_val = 1e5 * u.GeV
    Emax_val = 1e8 * u.GeV
    Emin_det_val = 1e5 * u.GeV

    diff_norm_val = 1.44e-14 * (1 / (u.GeV * u.m ** 2 * u.s))
    obs_time = 10 * u.year

    # Setup
    Parameter.clear_registry()
    index = Parameter(alpha, "index", fixed=False, par_range=(1.0, 4))
    L = Parameter(lumi, "luminosity", fixed=True, par_range=(0, 1e60))
    diffuse_norm = Parameter(
        diff_norm_val, "diffuse_norm", fixed=True, par_range=(0, np.inf)
    )
    Enorm = Parameter(Enorm_val, "Enorm", fixed=True)
    Emin = Parameter(Emin_val, "Emin", fixed=True)
    Emax = Parameter(Emax_val, "Emax", fixed=True)
    Emin_det = Parameter(Emin_det_val, "Emin_det", fixed=True)

    point_source = PointSource.make_powerlaw_source(
        "test", np.deg2rad(5) * u.rad, np.pi * u.rad, L, index, 0.5, Emin, Emax
    )

    my_sources = Sources()
    my_sources.add(point_source)
    my_sources.add_diffuse_component(diffuse_norm, Enorm.value)
    my_sources.add_atmospheric_component()

    f = my_sources.associated_fraction().value

    truths = {}
    diffuse_bg = my_sources.diffuse_component()
    truths["F_diff"] = diffuse_bg.flux_model.total_flux_int.value
    atmo_bg = my_sources.atmo_component()
    truths["F_atmo"] = atmo_bg.flux_model.total_flux_int.value
    truths["L"] = lumi.value
    truths["f"] = f
    truths["alpha"] = alpha

    outputs = {}
    outputs["F_diff"] = []
    outputs["F_atmo"] = []
    outputs["L"] = []
    outputs["f"] = []
    outputs["alpha"] = []

    for i in range(N):

        print("Run %i" % i)

        # Simulation
        sim = Simulation(my_sources, NorthernTracksDetectorModel, obs_time)
        sim.precomputation()
        sim.set_stan_filenames(atmo_sim_filename, main_sim_filename)
        sim.compile_stan_code(include_paths=include_paths)
        sim.run()

        lam = sim._sim_output.stan_variable("Lambda").values[0]
        sim_output = {}
        sim_output["Lambda"] = lam

        events = sim.events

        # Fit
        fit = StanFit(my_sources, NorthernTracksDetectorModel, events, obs_time)
        fit.precomputation()
        fit.set_stan_filename(fit_filename)
        fit.compile_stan_code(include_paths=include_paths)
        fit.run()

        # Store output
        outputs["F_diff"].append(
            np.mean(fit._fit_output.stan_variable("F_diff").values.T[0])
        )
        outputs["F_atmo"].append(
            np.mean(fit._fit_output.stan_variable("F_atmo").values.T[0])
        )
        outputs["L"].append(np.mean(fit._fit_output.stan_variable("L").values.T[0]))
        outputs["f"].append(np.mean(fit._fit_output.stan_variable("f").values.T[0]))
        outputs["alpha"].append(
            np.mean(fit._fit_output.stan_variable("alpha").values.T[0])
        )

        fit.check_classification(sim_output)

    print("time:", time.time() - start_time)

    return truths, outputs


results = Parallel(n_jobs=4, backend="loky")(delayed(run_fit)() for _ in range(4))

print(np.shape(results))

output_file = "output/sim_study_test.h5"

# Save
with h5py.File(output_file, "w") as f:

    for i, res in enumerate(results):
        folder = f.create_group("results_%i" % i)

        truths_folder = folder.create_group("truths")
        outputs_folder = folder.create_group("outputs")

        for key, value in res[0].items():
            truths_folder.create_dataset(key, data=value)

        for key, value in res[1].items():
            outputs_folder.create_dataset(key, data=value)
