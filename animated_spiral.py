import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

fig = plt.figure()
ax=fig.add_subplot(projection='3d')
line, = ax.plot([], [], [])
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(0,3)

n_points = 1001
tf = 3
f = 1
fps = int(np.ceil(n_points/tf))
t = np.linspace(0,tf,n_points)
frame_amount = int(n_points)
frame_speed = int(np.ceil(1000/fps))
print(fps)
X = 0.5*np.cos(2*np.pi*f*t)
Y = 0.5*np.sin(2*np.pi*f*t)

def animate(num):
    line.set_xdata(X[0:num])
    line.set_ydata(Y[0:num])
    line.set_3d_properties(t[0:num])
    # line.set_data(X[0:i],Y[0:i],t[0:i])
    return line,

ani=animation.FuncAnimation(fig, animate,
    frames=frame_amount,interval=frame_speed,repeat=False,blit=True)
# ani.save('spiral.mp4', fps=fps, dpi=120)
plt.show()

