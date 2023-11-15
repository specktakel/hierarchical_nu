import astropy.units as u
import numpy as np
from scipy import integrate
from scipy.optimize import root_scalar

Om = 0.3
Ol = 0.7
H0 = 70 * u.km / u.s / u.Mpc
c = 3e5 * u.km / u.s
DH = c / H0  # Mpc


def E(z):
    Omp = Om * (1 + z) ** 3
    return np.sqrt(Omp + Ol)


def hubble_factor(z):
    return H0 * E(z)


def comoving_distance(z):
    scale = lambda z: 1 / E(z)
    result, err = integrate.quad(scale, 0, z)

    return DH * result


def luminosity_distance(z):
    return (1 + z) * comoving_distance(z)


@u.quantity_input
def redshift(dL: u.Mpc):
    def z_of_dl(z, dL):
        return dL - luminosity_distance(z).to_value(u.Mpc)

    solution = root_scalar(
        z_of_dl, (dL.to_value(u.Mpc)), method="brentq", bracket=[1e-4, 10.0]
    )
    if solution.converged:
        return solution.root
    else:
        raise ValueError("No solution found.")
