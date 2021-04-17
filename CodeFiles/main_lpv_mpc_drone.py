import numpy as np
import matplotlib.pyplot as plt
import support_files_drone as sfd
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

support = sfd.SupportFilesDrone()
constants = support.constants

Ts = constants['Ts']
controlled_states = constants['controlled_states']
innerDyn_length = constants['innerDyn_length']
pos_x_y = constants['pos_x_y']    # chooses whether or not to make the body-frame x and y axis visualizations longer
sub_loop = constants['sub_loop']
if pos_x_y == 1:
    extension = 2.5
elif pos_x_y == 0:
    extension = 0
else:
    print('pos_x_y variable must be 1 or 0 in the __init__ method in the SupportFilesDrone class.')
    exit()

# generate reference signals
t = np.arange(0,100+Ts*innerDyn_length,Ts*innerDyn_length)
t_angles = np.arange(0,t[-1]+Ts,Ts)
t_ani = np.arange(0, t[-1]+Ts/sub_loop,Ts/sub_loop)
x_ref, x_dot_ref, x_dot_dot_ref, y_ref, y_dot_ref, y_dot_dot_ref, z_ref, z_dot_ref, z_dot_dot_ref, psi_ref = support.trajectory_generator(t)
plotl = len(t)

# create initial state vector
pt = 0
qt = 0
rt = 0
ut = 0
vt = 0
wt = 0
xt = 0
yt = -1
zt = 0
phit = 0
thetat = 0
psit = psi_ref[0]
states = np.array([ut,vt,wt,pt,qt,rt,xt,yt,zt,phit,thetat,psit])
statesTotal = [states]
statesTotal_ani = states[6:]

ref_angles_total = np.array([[phit, thetat, psit]])
velocityXYZ_total = np.array([[0,0,0]])

# initial propellor angular velocities
omega1 = 110*np.pi/3 # radians per second
omega2 = 110*np.pi/3
omega3 = 110*np.pi/3
omega4 = 110*np.pi/3
omega_total = omega1 + omega2 + omega3 + omega4

# initial control inputs
ct = constants['ct']
cq = constants['cq']
l = constants['l']

U1 = ct*(omega1**2 + omega2**2 + omega3**2 + omega4**2)
U2 = ct*l*(omega2**2 - omega4**2)
U3 = ct*l*(omega3**2 - omega1**2)
U4 = -cq*(omega1**2 - omega2**2 + omega3**2 - omega4**2)
UTotal = np.array([[U1,U2,U3,U4]])
omegas_bundle = np.array([[U1,U2,U3,U4]])
UTotal_ani = UTotal

########## BEGIN GLOBAL CONTROLLER ##########
for i_global in range(0, plotl-1):
    # implement the position controller (feedback linearization)
    phi_ref, theta_ref, U1 = support.pos_controller