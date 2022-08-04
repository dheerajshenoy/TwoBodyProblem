import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# universal gravitation constant
# G = 6.67408e-11 (m**3/kg/s**2)
G = 6.67259e-20  # (km**3/kg/s**2)

# time array
time = np.arange(0, 480, 0.5)

# body m1 initial conditions
m1 = 1e26  # mass (kg)
r10 = np.array([0, 0, 0])  # initial position (km)
v10 = np.array([0, 0, 0])  # initial velocity (km/s)

# body m2 initial conditions
m2 = 1e20  # mass (kg)
r20 = np.array([3000, 0, 0])  # initial position (km)
v20 = np.array([10, 20, 30])  # initial velocity (km/s)

fig = plt.figure()
ax = plt.axes(projection='3d')

# [X1 (0), Y1 (1), Z1 (2), X2 (3), Y2 (4), Z2 (5), 
#  VX1 (6), VY1 (7), VZ1 (8), VX2 (9), VY2 (10), VZ2 (11)]
y0 = np.concatenate((r10, r20, v10, v20))

def two_body_eqm(y, t, G, m1, m2):

    r21 = y[3:6] - y[:3]
    r21_mag = np.linalg.norm(r21)

    r_c = np.power(r21_mag, 3)

    k1 = G * m2 / r_c
    k2 = G * m1 / r_c
    c0 = y[6:12]
    c1 = r21 * k1
    c2 = -r21 * k2

    return np.concatenate((c0, c1, c2))

y = odeint(two_body_eqm, y0, time, args=(G, m1, m2))

xdata1 = []
ydata1 = []
zdata1 = []

xdata2 = []
ydata2 = []
zdata2 = []

for ys in y:
    Y1 = ys[:3]
    Y2 = ys[3:6]
    xdata1.append(Y1[0])
    ydata1.append(Y1[1])
    zdata1.append(Y1[2])

    xdata2.append(Y2[0])
    ydata2.append(Y2[1])
    zdata2.append(Y2[2])

line1, = ax.plot([r10[0]], [r10[1]], [r10[2]], lw=2, color='blue')
line2, = ax.plot([], [], [], lw=2, color='green')

def gg():
    i = 0
    # extract inertial positions of body 1 and body 2
    r1 = y[i][:3]
    r2 = y[i][3:6]

    # determine position of centre of mass
    rg = ((m1 * r1) + (m2 * r2)) / (m1 + m2)

    # position vector from m1 to m2
    r12 = r2 - r1

    # position vector from m1 to g
    r1g = rg - r1

    # position vector from g to m1
    rg1 = r1 - rg

    # position vector from g to m2
    rg2 = r2 - rg

    # save state history (yk = 0-11, rg = 12-14, r12=15-17, ...)
    #state_history.append(np.concatenate((yk, rg, r12, r1g, rg1, rg2), axis=None))

    x1, y1, z1 = r1[0], r1[1], r1[2]
    x2, y2, z2 = r2[0], r2[1], r2[2]

    rx, ry, rz = rg[0], rg[1], rg[2]

ax.set_xlim(-3000, 3000)
ax.set_ylim(-3000, 3000)
ax.set_zlim(-3000, 3000)



def animate(i):
    line1.set_data(xdata1[:i], ydata1[:i])
    line1.set_3d_properties(zdata1[:i])

    line2.set_data(xdata2[:i], ydata2[:i])
    line2.set_3d_properties(zdata2[:i])

    return (line1, line2)

anim = FuncAnimation(fig, animate, frames = len(y), interval = 10, blit = True)
plt.show()

