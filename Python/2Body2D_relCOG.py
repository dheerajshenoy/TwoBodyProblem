import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# universal gravitation constant
# G = 6.67408e-11 (m**3/kg/s**2)
G = 6.67259e-20  # (km**3/kg/s**2)

# time array
time = np.arange(0, 100, 0.2)

fig, ax = plt.subplots(1, 1)


m1 = 1e26  # mass (kg)
r10 = np.array([0, 0])  # initial position (km)
v10 = np.array([0, 0])  # initial velocity (km/s)
radius1 = 25

# body m2 initial conditions
m2 = 1e26  # mass (kg)
r20 = np.array([1500, 0])  # initial position (km)
#v20 = np.array([-30, -20])  # initial velocity (km/s)
v20 = np.array([-40, 40])  # initial velocity (km/s)
radius2 = 25

ax.set_xlim(-2000, 2000)
ax.set_ylim(-2000, 2000)

# [X1 (0), Y1 (1), X2 (2), Y2 (3), 
#  VX1 (4), VY1 (5), VX2 (6), VY2 (7)]
y0 = np.concatenate((r10, r20, v10, v20))

def two_body_eqm(y, t, G, m1, m2):
    r_g = (m1 * y[:2] + m2 * y[2:4])/(m1 + m2)

    r1 = y[:2] - r_g
    r2 = y[2:4] - r_g

    mu = G * (m1 + m2)
    mu1 = (m1/(m1 + m2))**3 * mu
    mu2 = (m2/(m1 + m2))**3 * mu

    c0 = y[4:8]
    c1 = -mu2 * r1/(np.linalg.norm(r1)**3)
    c2 = -mu1 * r2/(np.linalg.norm(r2)**3)

    return np.concatenate((c0, c1, c2))

y = odeint(two_body_eqm, y0, time, args=(G, m1, m2))

xdata1 = []
ydata1 = []
vxdata1 = []
vydata1 = []

xdata2 = []
ydata2 = []
vxdata2 = []
vydata2 = []

for ys in y:
    Y1 = ys[:2]
    Y2 = ys[2:4]
    Y3 = ys[4:]

    xdata1.append(Y1[0])
    ydata1.append(Y1[1])

    xdata2.append(Y2[0])
    ydata2.append(Y2[1])

line1, = ax.plot([], [], lw=1, color='blue', alpha=0.75 )
point1 = plt.Circle((0, 0), radius=radius1, color='blue', label = f"m1 = {m1} (kg)")

line2, = ax.plot([], [], lw=1, color='red', alpha=0.75 )
point2 = plt.Circle((0, 0), radius=radius2, color='green', label = f"m2 = {m2} (kg)")

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

def animate(i):

    line1.set_data(xdata1[:i], ydata1[:i])
    r10 = (xdata1[i], ydata1[i])
    point1.center = r10

    line2.set_data(xdata2[:i], ydata2[:i])
    r20 = (xdata2[i], ydata2[i])
    point2.center = r20
    return (line1, line2, point1, point2)

anim = FuncAnimation(fig, animate, frames = len(time), interval = 10, blit = False)
plt.show()
