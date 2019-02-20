import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bspline_ev import bspline_func_2d

p = 3 # spline degree
tx_orig = np.array([2.90998783, 4.03010253, 5.00991847, 6.96995941]) # knot sequence
ty_orig = np.array([-0.9,  0.1,  0.9]) # knot sequence
Nx = len(tx_orig)+p-1 # number of coefficients that need to be defined
Ny = len(ty_orig)+p-1 # number of coefficients that need to be defined
N = Nx * Ny


c = np.asarray([-3.58077707, -3.41361407, -3.67857551, -4.18870088, -4.33580191, -1.96090326,
 -1.97689,    -2.05099873, -2.14549395, -2.17675871, -0.34077015, -0.32737177,
 -0.30599357, -0.29709922, -0.29278352,  0.48041494,  0.94140028,  1.02319478,
  0.88547053,  0.95448815, -0.26299716,  0.99602738,  1.47548988,  1.31308221,
  1.42759428, -1.33184261,  0.47994404,  1.74747881,  1.44319947,  1.39107924])

c = c.reshape(Nx, Ny)
func = bspline_func_2d(tx_orig, ty_orig, p, c)

xaxis = np.linspace(tx_orig[0], tx_orig[-1], 100)
yaxis = np.linspace(ty_orig[0], ty_orig[-1], 100)

z = np.asarray([func.eval(tx, ty) for tx in xaxis for ty in yaxis])
z = z.reshape(len(xaxis), len(yaxis))

xx, yy = np.meshgrid(xaxis, yaxis)
xx=xx.T
yy=yy.T

fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_wireframe(xx, yy, z, rstride=5, cstride=5, color='blue')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlim(-4,3)
plt.show()
