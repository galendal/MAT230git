#%%
import numpy
import scipy
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def cart_pend(t, x, pend_mass, cart_Mass, L, g, d, u):
    """

    This function is to idenstify the our equations that link our states to the inputs

    """

    Sx = numpy.sin(x[2])
    Cx = numpy.cos(x[2])
    D = pend_mass * L * L * (cart_Mass + pend_mass * (1 - Cx**2))

    dx1_dt = x[1]
    dx2_dt = (1 / D) * (
        -(pend_mass**2) * L**2 * g * Cx * Sx
        + pend_mass * L**2 * (pend_mass * L * x[3] ** 2 * Sx - d * x[1])
    ) + pend_mass * L * L * (1 / D) * u
    dx3_dt = x[3]
    dx4_dt = (1 / D) * (
        (pend_mass + cart_Mass) * pend_mass * g * L * Sx
        - pend_mass * L * Cx * (pend_mass * L * x[3] ** 2 * Sx - d * x[1])
    ) - pend_mass * L * Cx * (1 / D) * u

    return [dx1_dt, dx2_dt, dx3_dt, dx4_dt]


m = 1
M = 5
L = 2
g = -10
d = 1
tspan = (1.0, 10.0)
x0 = [0, 0, numpy.pi, 0.5]
print("the shape of initial state vector is", numpy.shape(x0))
sol = solve_ivp(lambda t, x: cart_pend(t, x, m, M, L, g, d, -1), tspan, x0)
t = sol.t
y1, y2, y3, y4 = sol.y

print("Time points:", t)
print("Solution for y1:", y1)
print("Solution for y2:", y2)

plt.plot(t, y1, label="y1")
plt.plot(t, y2, label="y2")
plt.xlabel("Time")
plt.ylabel("State Variables")
plt.legend()
plt.grid(True)
plt.show()

#fig, ax = plt.subplots()
#(line,) = ax.plot([], [], "b")
#
#
#def init():
#    ax.set_xlim(-2, 2)
#    ax.set_ylim(-2, 2)
#    return (line,)
#
#
#def update(frame):
#    line.set_data(y1[:frame], y2[:frame])
#    return (line,)
#
#
#ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True)
#ani.save("simulation.gif", writer="Pillow")
#
#plt.show()


# Initialize figure and axis
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Initialize shapes
shape1 = plt.Circle((0, 0), 0.1, color='red')
shape2 = plt.Rectangle((0, 0), 0.2, 0.2, color='blue')

# Update function to compute new positions
def update(frame):
    theta = sol.y[0, frame]
    x = sol.y[2, frame]
    shape1.center = (numpy.sin(theta), numpy.cos(theta))
    shape2.set_xy([x - 0.1, -0.1])
    return shape1, shape2

# Animation function
def animate(frame):
    ax.clear()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.add_patch(shape1)
    ax.add_patch(shape2)
    return update(frame)

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=sol.y.shape[1], interval=100)

# Display the animation
plt.show()

# Convert animation to HTML format
html_anim = anim.to_jshtml()

# Save animation to an HTML file
with open("animation.html", "w") as f:
    f.write(html_anim)

# Display the animation
plt.close()
# %%
