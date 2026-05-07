# Plot test functions
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

plt.figure(1, figsize=(15,8))
plt.clf()

# ----------------------------------------------------------------------------- #
x1 = np.linspace(-1.5, 1.5, 1000)
y1 = np.linspace(-1.5, 1.5, 1000)
X1, Y1 = np.meshgrid(x1, y1)
f1_xy = (1 - X1)**2 + 100*(Y1 - X1**2)**2

plt.subplot(2,3,1)
plt.contourf(X1, Y1, f1_xy, locator=ticker.LogLocator())
#fig1 = plt.figure(1)
#ax1 = fig1.add_subplot(projection = '3d')
#ax1.plot_surface(X1, Y1, f1_xy)
#plt.tight_layout()
plt.plot(1.0, 1.0, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test_001: Rosenbrock')
plt.colorbar()

# ----------------------------------------------------------------------------- #
x2 = np.linspace(-10.0, 0.0, 1000)
y2 = np.linspace(-6.5, 0.0, 1000)
X2, Y2 = np.meshgrid(x2, y2)
f2_xy = np.sin(Y2)*np.exp((1 - np.cos(X2))**2)\
    + np.cos(X2)*np.exp((1 - np.sin(Y2))**2) + (X2 - Y2)**2

plt.subplot(2,3,2)
plt.contourf(X2, Y2, f2_xy)
#fig2 = plt.figure(2)
#ax2 = fig2.add_subplot(projection = '3d')
#ax2.plot_surface(X2, Y2, f2_xy)
#plt.tight_layout()
plt.plot(-3.13024, -1.58218, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test_002: Mishra Bird')
plt.colorbar()

# ----------------------------------------------------------------------------- #
x3 = np.linspace(-2.25, 2.25, 1000)
y3 = np.linspace(-2.5, 1.75, 1000)
X3, Y3 = np.meshgrid(x3, y3)
f3_xy = -1*(np.cos((X3 - 0.1)*Y3)**2) - X3*np.sin(3*X3 + Y3)

plt.subplot(2,3,3)
plt.contourf(X3, Y3, f3_xy)
#fig3 = plt.figure(3)
#ax3 = fig3.add_subplot(projection = '3d')
#ax3.plot_surface(X3, Y3, f3_xy)
#plt.tight_layout()
plt.plot(2.03587, 1.13376, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test_003: Townsend')
plt.colorbar()

# ----------------------------------------------------------------------------- #
x4 = np.linspace(-10.0, 10.0, 1000)
y4 = np.linspace(-10.0, 10.0, 1000)
X4, Y4 = np.meshgrid(x4, y4)
f4_xy = 0.26*(X4**2 + Y4**2) - 0.48*X4*Y4

plt.subplot(2,3,4)
plt.contourf(X4, Y4, f4_xy, locator=ticker.LogLocator())
#fig4 = plt.figure(4)
#ax4 = fig4.add_subplot(projection = '3d')
#ax4.plot_surface(X4, Y4, f4_xy)
#plt.tight_layout()
plt.plot(-3.6596e-12, -3.66993e-12, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test_004: Matyas')
plt.colorbar()

# ----------------------------------------------------------------------------- #
x5 = np.linspace(-10.0, 10.0, 1000)
y5 = np.linspace(-10.0, 10.0, 1000)
X5, Y5 = np.meshgrid(x5, y5)
f5_xy = (X5 + 2*Y5 - 7)**2 + (2*X5 + Y5 - 5)**2

plt.subplot(2,3,5)
plt.contourf(X5, Y5, f5_xy, locator=ticker.LogLocator())
#fig5 = plt.figure(5)
#ax5 = fig4.add_subplot(projection = '3d')
#ax5.plot_surface(X5, Y5, f5_xy)
#plt.tight_layout()
plt.plot(1, 3, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test_005: Booth')
plt.colorbar()

# ----------------------------------------------------------------------------- #
x6 = np.linspace(-2.0, 2.0, 1000)
y6 = np.linspace(-3.0, 1.0, 1000)
X6, Y6 = np.meshgrid(x6, y6)
f6_xy = (1 + (X6 + Y6 + 1)**2*(19 - 14*X6 + 3*X6**2 - 14*Y6 + 6*X6*Y6 + 3*Y6**2))\
    *(30 + (2*X6 - 3*Y6)**2*(18 - 32*X6 + 12*X6**2 + 48*Y6 - 36*X6*Y6 + 27*Y6**2))

plt.subplot(2,3,6)
plt.contourf(X6, Y6, f6_xy, locator=ticker.LogLocator())
#fig6 = plt.figure(6)
#ax6 = fig6.add_subplot(projection = '3d')
#ax6.plot_surface(X6, Y6, f6_xy)
#plt.tight_layout()
plt.plot(1.56743e-05, -0.999994, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('test_006: Goldstein-Price')
plt.colorbar()

# ----------------------------------------------------------------------------- #

plt.tight_layout()
plt.show()