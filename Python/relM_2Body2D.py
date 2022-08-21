import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# universal gravitation constant
# G = 6.67408e-11 (m**3/kg/s**2)
G = 6.67259e-20  # (km**3/kg/s**2)

# time array
time = np.arange(0, 480, 0.25)
fig, ax = plt.subplots(1, 1)

m1 = 1e26  # mass (kg)
r10 = np.array([0, 0])  # initial position (km)
v10 = np.array([0, 0])  # initial velocity (km/s)
radius1 = 25

# body m2 initial conditions
m2 = 1e26  # mass (kg)
r20 = np.array([1000, 1000])  # initial position (km)
v20 = np.array([-30, 10])  # initial velocity (km/s)
radius2 = 25
ax.set_xlim(-1500, 1500)
ax.set_ylim(-500, 1500)

# [X1 (0), Y1 (1), VX1 (2), VY1 (3)]
y0 = np.concatenate((r20, v20))

def two_body_eqm(y, t, G, m1, m2):

    r_mag = np.linalg.norm(y[:2])
    r_c = np.power(r_mag, 3)

    mu = G * (m1 + m2)

    c0 = y[2:4]
    c1 = -mu / r_c * y[:2]

    return np.concatenate((c0, c1))

y = odeint(two_body_eqm, y0, time, args=(G, m1, m2))

xdata1 = []
ydata1 = []
vxdata1 = []
vydata1 = []

for ys in y:
    Y1 = ys[:2]
    Y2 = ys[2:4]

    xdata1.append(Y1[0])
    ydata1.append(Y1[1])

line1, = ax.plot([], [], lw=1, color='blue', alpha=0.75 )
point1 = plt.Circle((0, 0), radius=radius1, color='blue', label = f"m1 = {m1} (kg)")
point2 = plt.Circle((0, 0), radius=radius2, color='red', label = f"m2 = {m2} (kg)")

plt.title(f"Relative motion of m_1 with respect to m_2\nv1 = {v10} (km/s) v2 = {v20} (km/s)")

ax.add_artist(point1)
ax.add_artist(point2)

ax.legend()

def func():
    global r10
    global r20
    for i in range(len(time)):

        line1.set_data(xdata1[:i], ydata1[:i])
        r10 = (xdata1[i], ydata1[i])
        point1.center = r10
        plt.savefig("Anim/%s"%str(i))

ax.set_aspect('equal')

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

def animate(i):

    line1.set_data(xdata1[:i], ydata1[:i])
    r10 = (xdata1[i], ydata1[i])
    point1.center = r10
    return (line1, point1, point2)

anim = FuncAnimation(fig, animate, frames = len(time), interval = 10, blit = False)
plt.show()
