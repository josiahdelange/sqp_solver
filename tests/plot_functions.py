# Plot test functions
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

plt.figure(1, figsize=(15,5))
plt.clf()

# ----------------------------------------------------------------------------- #
x1 = np.linspace(-1.5, 1.5, 1000)
y1 = np.linspace(-1.5, 1.5, 1000)
X1, Y1 = np.meshgrid(x1, y1)
f1_xy = (1 - X1)**2 + 100*(Y1 - X1**2)**2

plt.subplot(1,3,1)
plt.contourf(X1, Y1, f1_xy)
#fig1 = plt.figure(1)
#ax1 = fig1.add_subplot(projection = '3d')
#ax1.plot_surface(X1, Y1, f1_xy)
#plt.tight_layout()
plt.plot(1.0, 1.0, 'rx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rosenbrock')
plt.colorbar()

# ----------------------------------------------------------------------------- #
x2 = np.linspace(-10.0, 0.0, 1000)
y2 = np.linspace(-6.5, 0.0, 1000)
X2, Y2 = np.meshgrid(x2, y2)
f2_xy = np.sin(Y2)*np.exp((1 - np.cos(X2))**2)\
    + np.cos(X2)*np.exp((1 - np.sin(Y2))**2) + (X2 - Y2)**2

plt.subplot(1,3,2)
plt.contourf(X2, Y2, f2_xy)
#fig2 = plt.figure(2)
#ax2 = fig2.add_subplot(projection = '3d')
#ax2.plot_surface(X2, Y2, f2_xy)
#plt.tight_layout()
plt.plot(-3.13024, -1.58218, 'rx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Mishra Bird')
plt.colorbar()

# ----------------------------------------------------------------------------- #
x3 = np.linspace(-2.25, 2.25, 1000)
y3 = np.linspace(-2.5, 1.75, 1000)
X3, Y3 = np.meshgrid(x3, y3)
f3_xy = -1*(np.cos((X3 - 0.1)*Y3)**2) - X3*np.sin(3*X3 + Y3)

plt.subplot(1,3,3)
plt.contourf(X3, Y3, f3_xy)
#fig3 = plt.figure(3)
#ax3 = fig3.add_subplot(projection = '3d')
#ax3.plot_surface(X3, Y3, f3_xy)
#plt.tight_layout()
plt.plot(2.03587, 1.13376, 'rx')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Townsend')
plt.colorbar()

# ----------------------------------------------------------------------------- #

plt.tight_layout()
plt.show()