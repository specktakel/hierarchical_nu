
import astropy.units as u
import numpy as np
from scipy import integrate

Om = 0.3
Ol = 0.7
H0 = 70 * u.km / u.s / u.Mpc
c = 3E5 * u.km / u.s
DH = c / H0  # Mpc


def E(z):
    Omp = Om * (1 + z)**3
    return np.sqrt(Omp + Ol)


def hubble_factor(z):
    return H0 * E(z)


def comoving_distance(z):
    scale = lambda z: 1 / E(z)
    result, err = integrate.quad(scale, 0, z)

    return DH * result


def luminosity_distance(z):
    return (1 + z) * comoving_distance(z)
