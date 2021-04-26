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
    phi_ref, theta_ref, U1 = support.pos_controller(x_ref[i_global+1], x_dot_ref[i_global+1], x_dot_dot_ref[i_global+1], y_ref[i_global+1], y_dot_ref[i_global+1], y_dot_dot_ref[i_global+1], z_ref[i_global+1], z_dot_ref[i_global+1], z_dot_dot_ref[i_global+1], psi_ref[i_global+1], states)
    phi_ref = phi_ref*np.ones(innerDyn_length+1).T
    theta_ref = theta_ref*np.ones(innerDyn_length+1).T
    psi_ref = np.zeros(innerDyn_length+1).T
    for yaw_step in range(0, innerDyn_length+1):
        psi_ref[yaw_step] = psi_ref[i_global] + (psi_ref[i_global+1]-psi_ref[i_global])/(Ts*innerDyn_length)*Ts*yaw_step
    temp_angles = np.concatenate((phi_ref[1:],theta_ref[1:],psi_ref[1:]),axis=1)
    ref_angles_total = np.concatenate((ref_angles_total,temp_angles),axis=0)
    # create a reference vector
    refSignals = np.zeros(len(phi_ref)*controlled_states)

    # Build up the reference signal vector
    # refSignal = [phi_ref_0, theta_ref_0, psi_ref_0, phi_ref_1, theta_ref_1, ...]
    k = 0
    for i in range(0,len(refSignals), controlled_states):
        refSignals[i] = phi_ref[k]
        refSignals[i+1] = theta_ref[k]
        refSignals[i+2] = psi_ref[k]
        k += 1

    # Initialize the controller - simulation loops
    hz = support.constants['hz']
    k = 0

    for i in range(0, innerDyn_length):
        # Generate the discrete state space matrices
        Ad, Bd, Cd, Dc, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot = support.LPV_cont_discrete(states, omega_total)
        x_dot = x_dot.T
        y_dot = y_dot.T
        z_dot = z_dot.T
        temp_velocityXYZ = np.concatenate(([[x_dot],[y_dot],[z_dot]]),axis=1)
        velocityXYZ_total = np.concatenate((velocityXYZ_total, temp_velocityXYZ),axis=0)
        # Generate the augmented current state and reference vector (9x1)
        x_aug_t = np.concatenate(([phi,phi_dot,theta,theta_dot,psi,psi_dot],[U2,U3,U4]),axis=0)
        # Ts = 0.1s
        # From the refSignals vector, only extract the reference values from  [current sample (NOW) + Ts]
        # to [NOW + horizon period (hz)]
        # EXAMPLE: t_now is 3 seconds, hz = 15 samples. From the refSignals vector, you take the elements
        # r = [phi_ref_3.1, theta_ref_3.1, psi_ref_3.1, phi_ref_3.2, ...]
        k += controlled_states
        if k + controlled_states*hz <= len(refSignals):
            r = refSignals[k:k+controlled_states*hz]
        else:
            r = refSignals[k:]
            hz -= 1

        # Generate the compact simplification matrices for the cost function
        Hdb, Fdbt, Cdb, Adc = support.mpc_simplification(Ad, Bd, Cd, Dd, hz)
        ft = np.concatenate((x_aug_t.T,r),axis=0)@Fdbt