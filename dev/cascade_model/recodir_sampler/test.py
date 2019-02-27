#!/usr/bin/env python

from vMF_sampler import VonMisesFisher
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from uniform_sampler import transform_to_spherical


def rand_uniform_sphere(npts, return_in_coordinate_sys='cartesian'):

    # (zenith, azimuth) =  (theta, phi) 

    phi = 2.0 * np.pi * np.random.uniform(0.0, 1.0, npts)
    theta = np.arccos(2.0 * np.random.uniform(0.0, 1.0, npts) - 1.0)
 
    V = np.column_stack([theta, phi])
    if return_in_coordinate_sys == 'spherical':
        return V

    elif return_in_coordinate_sys == 'cartesian':
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta) 
        V = np.column_stack([x, y, z])
        return V


def test_plot_on_sphere(X):

    # Points must be in cartesian coordinates

    color = ['r', 'g', 'b', 'k', 'y']

    s=[20,3,3]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        x = X[i][:, 0]
        y = X[i][:, 1]
        z = X[i][:, 2]

        ax.scatter(x, y, z, c=color[i], marker='o', s=s[i])
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0) 

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    x = X[0][:, 0]
    y = X[0][:, 1]
    z = X[0][:, 2]

    ax.scatter(x, y, z, c=color[0], marker='o', s=40, zorder=5)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show(ax)


vMF = VonMisesFisher()
#mu = rand_uniform_sphere(1, return_in_coordinate_sys='cartesian')
mu = np.array([1.0, -0.05, 0.0])
mu = np.array([mu / np.linalg.norm(mu)])
print mu
print transform_to_spherical(mu)


## kappa ~ 2.3 / sigma**2
err = 10 # in degree
err = err / 180. * np.pi
var = err ** 2
kappa = 2.3 / var


data_clustered = vMF.randomize_about_point(mu, kappa=kappa, num_samples=500)
data_unif = rand_uniform_sphere(2000, return_in_coordinate_sys='cartesian')
#data_unif = rand_uniform_sphere(1, return_in_coordinate_sys='cartesian')
test_plot_on_sphere([mu, data_clustered, data_unif])

reco_dir = []



#print transform_to_spherical(data_clustered)






