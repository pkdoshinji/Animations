#!/usr/bin/env python3

'''
Animated simulation of the double elastic pendulum

Author: M. Patrick Kelly
Last updated: 11-30-2020

Adapted from https://scipython.com/blog/the-spring-pendulum/ and
https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

'''

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.patches import Circle

# Gravitational acceleration (m/s^2)
G = 9.81

class DoubleElastic:

# Python class for the double elastic pendulum.

# Initial state is [a, b, alpha, beta, a_dot, b_dot, alpha_dot, beta_dot],
# where a and b are the respective lengths of the pendulum arms, alpha and
# beta are the angles between each arm and the vertical, and a_dot etc. are
# the time derivatives of these quantities. 

    def __init__(self,
                 init_state=[4, 4, 3*np.pi/4, np.pi/2, 0, 0, 0, 0],
                 L1=4.0,  # equilibrium length of arm 1 (m)
                 L2=4.0,  # equilibrium length of arm 2 (m)
                 K1=35,   # spring constant 1 (kg/s^2)
                 K2=35,  # spring constant 2 (kg/s^2)
                 M1=8.0,  # mass of bob 1 (in kg)
                 M2=8.0,  # mass of bob 2 (in kg)
                 dt=0.02,
                 color='r', # color of pendulum trail
                 trail_secs=2.0, # duration (seconds) of trail
                 origin=(0, 0)): # coordinates of pendulum anchor
            self.origin = origin
            self.state = np.asarray(init_state, dtype='float')
            self.position = self.get_position()
            self.color = color
            self.x2_trajectory = np.array(self.position[0][2])
            self.y2_trajectory = np.array(self.position[1][2])
            self.params = (L1, L2, K1, K2, M1, M2)
            self.dt = dt
            self.max_trail = int(trail_secs / dt)


    def get_position(self):
    #Convert the state array to Cartesian coordinates
        x = np.cumsum([self.origin[0],
                       self.state[0] * sin(self.state[2]),
                       self.state[1] * sin(self.state[3])])
        y = np.cumsum([self.origin[1],
                       -self.state[0] * cos(self.state[2]),
                       -self.state[1] * cos(self.state[3])])
        return (x, y)

    def dstate_dt(self, state, t):
    # Compute the derivative of the given state
    # These are the equations of motion for the double elastic pendulum
        (L1, L2, K1, K2, M1, M2) = self.params

        dydx = np.zeros_like(state)

        del_ = state[2] - state[3]
        dydx[0] = state[4]
        dydx[1] = state[5]
        dydx[2] = state[6]
        dydx[3] = state[7]
        dydx[4] = (state[0] * state[6] ** 2) + G * (np.cos(state[2])) \
                  - (K1 / M1) * (state[0] - L1) + (K2 / M1) * (state[1] - L2) \
                  * np.cos(del_)
        dydx[5] = (state[1] * state[7] ** 2) - (K2 / M2) * (state[1] - L2) \
                  - (K2 / M1) * (state[1] - L2) + (K1 / M1) * (state[0] - L1) \
                  * np.cos(del_)
        dydx[6] = (-1 / state[0]) * (2 * state[4] * state[6] + G * np.sin(state[2]) \
                  + (K2 / M1) * (state[1] - L2) * np.sin(del_))
        dydx[7] = (-1 / state[1]) * (2 * state[5] * state[7] - (K1 / M1) * \
                  (state[0] - L1) * np.sin(del_))

        return dydx

    def make_plot(self, i):
        # Plot the pendulum arms (springs)
        self.plot_spring(self.position[0][0], self.position[1][0], self.state[2], self.state[0])
        self.plot_spring(self.position[0][1], self.position[1][1], self.state[3], self.state[1])

        # Circles representing the anchor points and the bobs
        r = 0.4 # Radius of each bob
        self.c0 = Circle((self.position[0][0], self.position[1][0]), r, fc='k', zorder=10)
        self.c1 = Circle((self.position[0][1], self.position[1][1]), r, fc=self.color, ec=self.color, zorder=10)
        self.c2 = Circle((self.position[0][2], self.position[1][2]), r, fc=self.color, ec=self.color, zorder=10)
        ax.add_patch(self.c0)
        ax.add_patch(self.c1)
        ax.add_patch(self.c2)

        self.plot_trail(i)

        # Set the bounds of the axes
        ax.set_xlim(-40, 40)
        ax.set_ylim(-35, 20)
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off')

    def plot_spring(self, x_origin, y_origin, angle, length):
        rs, ns = 0.5, 15 # Spring turn radius, number of turns
        Ns = 1000 # Number of data points for the helix
        ipad1, ipad2 = 100, 150 # Padding (no coils) near anchor and bob
        w = np.linspace(0, length, Ns)
        xp = np.zeros(Ns) # Set up the helix along the x-axis...
        xp[ipad1:-ipad2] = rs * np.sin(2 * np.pi * ns * w[ipad1:-ipad2] / length)
        # ...then rotate it to align with  the pendulum and plot
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        xs, ys = - R @ np.vstack((xp, w))
        ax.plot(x_origin + xs, y_origin + ys, c='k', lw=2)

    def plot_trail(self, i):
        # Trail is divided into ns segments and plotted as fading line.

        ns = 15 # Number of segments in the trail
        s = self.max_trail // ns

        for j in range(ns):
            imin = i - (ns - j) * s
            if imin < 0:
                continue
            imax = imin + s + 1

            # The fading looks better if we square the fractional length
            # along the trail.
            alpha = (j / ns) ** 2

            # Plot the trail
            ax.plot(self.x2_trajectory[imin:imax], self.y2_trajectory[imin:imax],
                    c=self.color, solid_capstyle='butt', lw=2, alpha=alpha)

    def step(self):
        # Execute one time step of length dt and update state
        self.state = odeint(self.dstate_dt, self.state, [0, self.dt])[1]
        self.position = self.get_position() # Update Cartesian coordinates

        # Append new x2, y2 to trajectory (for plotting the trail)
        self.x2_trajectory = np.append(self.x2_trajectory, self.position[0][2])
        self.y2_trajectory = np.append(self.y2_trajectory, self.position[1][2])


tmax, dt = 20, 0.04#0.04 #  Running time of animation, increment interval
t = np.arange(0, tmax + dt, dt) # Time array
pendulum1 = DoubleElastic(init_state=[4, 4, 3 * np.pi / 4, np.pi / 2, 0, 0, 0, 0], dt=dt)
pendulum2 = DoubleElastic(init_state=[15, 15, -2 * np.pi / 3, -np.pi / 2, 0, 0, 0,0], color='limegreen', dt=dt)
pendulum3 = DoubleElastic(init_state=[17, 10, 0.2, 0.22, 0, 0, 0,0], color='fuchsia', dt=dt)
pendulum4 = DoubleElastic(init_state=[10, 13, -4 * np.pi / 5, np.pi, 0, 0, 0,0], K1=75, K2=75,color='mediumblue', dt=dt)
pendulum5 = DoubleElastic(init_state=[12, 15, 2.3 * np.pi / 3, -np.pi / 3, 0, 0, 0,0], color='gold', dt=dt)
pendulum6 = DoubleElastic(init_state=[14, 8, -np.pi / 5, -np.pi / 5, 0, 0, 0,0], K1=100, K2=100, color='dodgerblue', dt=dt)
pendulum7 = DoubleElastic(init_state=[17, 14, -2 * np.pi / 7, -np.pi / 5, 0, 0, 0,0], K1=80, K2=80,color='firebrick', dt=dt)
pendulum8 = DoubleElastic(init_state=[22, 23, -5 * np.pi / 7, -np.pi / 2, 0, 0, 0,0], color='darkcyan', dt=dt)

fps = 25 # Frames per second
di = int(1/fps/dt)
fig = plt.figure(figsize=(8.33333333, 6.25), dpi=72)
ax = fig.add_subplot(111)

# Plot current state of pendulum objects
for i in range(0, t.size, di):
    print(i // di, '/', t.size // di)
    pendulum1.make_plot(i)
    pendulum1.step()
    pendulum2.make_plot(i)
    pendulum2.step()
    pendulum3.make_plot(i)
    pendulum3.step()
    pendulum4.make_plot(i)
    pendulum4.step()
    pendulum5.make_plot(i)
    pendulum5.step()
    pendulum6.make_plot(i)
    pendulum6.step()
    pendulum7.make_plot(i)
    pendulum7.step()
    pendulum8.make_plot(i)
    pendulum8.step()

    # Save image frames for conversion to GIF
    plt.savefig('frames/_testimg{:04d}.png'.format(i // di), dpi=72)

    # Clear the Axes ready for the next image frame.
    plt.cla()
