import platform
print("Python " + platform.python_version())
import numpy as np
print("Numpy " + np.__version__)
import matplotlib
print("Matplotlib " + matplotlib.__version__)
import numpy as np
import matplotlib.pyplot as plt
import support_files_drone as sfd
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from rotation_matrices import RotationMatrix

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

sim_version = constants['sim_version']
if sim_version == 1:
    pass
elif sim_version == 2:
    pass
else:
    print('Sim version must be either 1 or 2.')
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
statesTotal_ani = [states[6:]]

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
    phi_ref, theta_ref, U1=support.pos_controller(x_ref[i_global+1], x_dot_ref[i_global+1], x_dot_dot_ref[i_global+1], y_ref[i_global+1], y_dot_ref[i_global+1], y_dot_dot_ref[i_global+1], z_ref[i_global+1], z_dot_ref[i_global+1], z_dot_dot_ref[i_global+1], psi_ref[i_global+1], states)
    Phi_ref=np.transpose([phi_ref*np.ones(innerDyn_length+1)])
    Theta_ref=np.transpose([theta_ref*np.ones(innerDyn_length+1)])

    # Make Psi_ref increase continuosly in a linear fashion per outer loop
    Psi_ref=np.transpose([np.zeros(innerDyn_length+1)])
    for yaw_step in range(0, innerDyn_length+1):
        Psi_ref[yaw_step]=psi_ref[i_global]+(psi_ref[i_global+1]-psi_ref[i_global])/(Ts*innerDyn_length)*Ts*yaw_step

    temp_angles=np.concatenate((Phi_ref[1:len(Phi_ref)],Theta_ref[1:len(Theta_ref)],Psi_ref[1:len(Psi_ref)]),axis=1)
    ref_angles_total=np.concatenate((ref_angles_total,temp_angles),axis=0)
    # Create a reference vector
    refSignals=np.zeros(len(Phi_ref)*controlled_states)

    # Build up the reference signal vector:
    # refSignal = [Phi_ref_0, Theta_ref_0, Psi_ref_0, Phi_ref_1, Theta_ref_2, Psi_ref_2, ... etc.]
    k=0
    for i in range(0,len(refSignals),controlled_states):
        refSignals[i]=Phi_ref[k]
        refSignals[i+1]=Theta_ref[k]
        refSignals[i+2]=Psi_ref[k]
        k=k+1
    # # implement the position controller (feedback linearization)
    # phi_ref, theta_ref, U1 = support.pos_controller(x_ref[i_global+1], x_dot_ref[i_global+1], x_dot_dot_ref[i_global+1], y_ref[i_global+1], y_dot_ref[i_global+1], y_dot_dot_ref[i_global+1], z_ref[i_global+1], z_dot_ref[i_global+1], z_dot_dot_ref[i_global+1], psi_ref[i_global+1], states)
    # phi_ref = np.transpose([phi_ref*np.ones(innerDyn_length+1)])
    # theta_ref = np.transpose([theta_ref*np.ones(innerDyn_length+1)])
    # psi_ref = np.transpose([np.zeros(innerDyn_length+1)])
    # for yaw_step in range(0, innerDyn_length+1):
    #     psi_ref[yaw_step] = psi_ref[i_global] + (psi_ref[i_global+1]-psi_ref[i_global])/(Ts*innerDyn_length)*Ts*yaw_step
    # temp_angles = np.concatenate((phi_ref[1:],theta_ref[1:],psi_ref[1:]),axis=1)
    # ref_angles_total = np.concatenate((ref_angles_total,temp_angles),axis=0)
    # # create a reference vector
    # refSignals = np.zeros(len(phi_ref)*controlled_states)

    # # Build up the reference signal vector
    # # refSignal = [phi_ref_0, theta_ref_0, psi_ref_0, phi_ref_1, theta_ref_1, ...]
    # k = 0
    # for i in range(0,len(refSignals), controlled_states):
    #     refSignals[i] = phi_ref[k]
    #     refSignals[i+1] = theta_ref[k]
    #     refSignals[i+2] = psi_ref[k]
    #     k += 1

    # Initialize the controller - simulation loops
    hz = support.constants['hz']
    k = 0

    for i in range(0, innerDyn_length):
        # Generate the discrete state space matrices
        Ad, Bd, Cd, Dd, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot = support.LPV_cont_discrete(states, omega_total)
        x_dot = np.array([x_dot]).reshape(-1,)
        y_dot = np.array([y_dot]).reshape(-1,)
        z_dot = np.array([z_dot]).reshape(-1,)
        temp_velocityXYZ = np.concatenate(([[x_dot],[y_dot],[z_dot]]),axis=1)
        velocityXYZ_total = np.concatenate((velocityXYZ_total, temp_velocityXYZ),axis=0)
        # Generate the augmented current state and reference vector (9x1)
        x_aug_t = np.transpose([np.concatenate(([phi,phi_dot,theta,theta_dot,psi,psi_dot],[U2,U3,U4]),axis=0)])
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
        ft = np.concatenate((x_aug_t.T[0],r),axis=0)@Fdbt
        du = -np.linalg.inv(Hdb)@Fdbt.T

        # Update the control inputs
        U2 += du[0][0]

        UTotal = np.concatenate((UTotal,np.array([[U1,U2,U3,U4]])),axis=0)

        # Compute the new omegas based on the new U values
        U1C = U1/ct
        U2C = U2/(ct*l)
        U3C = U3/(ct*l)
        U4C = U4/cq

        UC_vector = np.zeros((4,1))
        UC_vector[0,0] = U1C
        UC_vector[1,0] = U2C
        UC_vector[2,0] = U3C
        UC_vector[3,0] = U4C

        omega_matrix = np.zeros((4,4))
        omega_matrix[0,:] = 1
        omega_matrix[1,1] = 1
        omega_matrix[1,3] = -1
        omega_matrix[2,0] = -1
        omega_matrix[2,2] = 1
        omega_matrix[3,:] = 1
        omega_matrix[3,0] = -1
        omega_matrix[3,2] = -1

        omega_matrix_inv = np.linalg.inv(omega_matrix)
        omegas_vector = omega_matrix_inv@UC_vector

        omega4P2 = omegas_vector[0,0]
        omega3P2 = omegas_vector[1,0]
        omega2P2 = omegas_vector[2,0]
        omega1P2 = omegas_vector[3,0]

        if omega1P2<=0 or omega2P2<=0 or omega3P2<=0 or omega4P2<=0:
            print('You cannot take the square root of a negative number.')
            print('The trajectory may be too chaotic or it may have discontinuities in the path.')
            print('Try a smoother trajectory. Other causes may be variables such as Ts, hz, innerDyn_length,')
            print('or the poles px, py, pz.')
            exit()
        else:
            omega1 = np.sqrt(omega1P2)
            omega2 = np.sqrt(omega2P2) 
            omega3 = np.sqrt(omega3P2) 
            omega4 = np.sqrt(omega4P2)  

        omegas_bundle = np.concatenate((omegas_bundle, np.array([[omega1,omega2,omega3,omega4]])),axis=0)
        
        # Compute the new total omega
        omega_total = omega1 - omega2 + omega3 - omega4
        # Compute new states in the open loop system (interval Ts/10)
        states, states_ani, U_ani = support.open_loop_new_states(states, omega_total, U1, U2, U3, U4)
        statesTotal = np.concatenate((statesTotal, [states]),axis=0)
        statesTotal_ani=np.concatenate((statesTotal_ani,states_ani),axis=0)
        UTotal_ani=np.concatenate((UTotal_ani,U_ani),axis=0)

################################ ANIMATION LOOP ###############################
if max(y_ref)>=max(x_ref):
    max_ref=max(y_ref)
else:
    max_ref=max(x_ref)

if min(y_ref)<=min(x_ref):
    min_ref=min(y_ref)
else:
    min_ref=min(x_ref)

statesTotal_x=statesTotal_ani[:,0]
statesTotal_y=statesTotal_ani[:,1]
statesTotal_z=statesTotal_ani[:,2]
statesTotal_phi=statesTotal_ani[:,3]
statesTotal_theta=statesTotal_ani[:,4]
statesTotal_psi=statesTotal_ani[:,5]
UTotal_U1=UTotal_ani[:,0]
UTotal_U2=UTotal_ani[:,1]
UTotal_U3=UTotal_ani[:,2]
UTotal_U4=UTotal_ani[:,3]
frame_amount=int(len(statesTotal_x))
length_x=max_ref*0.15 # Length of one half of the UAV in the x-direction (Only for the animation purposes)
length_y=max_ref*0.15 # Length of one half of the UAV in the y-direction (Only for the animation purposes)

def update_plot(num):

    R_x=np.array([[1, 0, 0],[0, np.cos(statesTotal_phi[num]), -np.sin(statesTotal_phi[num])],[0, np.sin(statesTotal_phi[num]), np.cos(statesTotal_phi[num])]])
    R_y=np.array([[np.cos(statesTotal_theta[num]),0,np.sin(statesTotal_theta[num])],[0,1,0],[-np.sin(statesTotal_theta[num]),0,np.cos(statesTotal_theta[num])]])
    R_z=np.array([[np.cos(statesTotal_psi[num]),-np.sin(statesTotal_psi[num]),0],[np.sin(statesTotal_psi[num]),np.cos(statesTotal_psi[num]),0],[0,0,1]])
    R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))

    drone_pos_body_x=np.array([[length_x+extension],[0],[0]])
    drone_pos_inertial_x=np.matmul(R_matrix,drone_pos_body_x)

    drone_pos_body_x_neg=np.array([[-length_x],[0],[0]])
    drone_pos_inertial_x_neg=np.matmul(R_matrix,drone_pos_body_x_neg)

    drone_pos_body_y=np.array([[0],[length_y+extension],[0]])
    drone_pos_inertial_y=np.matmul(R_matrix,drone_pos_body_y)

    drone_pos_body_y_neg=np.array([[0],[-length_y],[0]])
    drone_pos_inertial_y_neg=np.matmul(R_matrix,drone_pos_body_y_neg)

    drone_body_x.set_xdata([statesTotal_x[num]+drone_pos_inertial_x_neg[0][0],statesTotal_x[num]+drone_pos_inertial_x[0][0]])
    drone_body_x.set_ydata([statesTotal_y[num]+drone_pos_inertial_x_neg[1][0],statesTotal_y[num]+drone_pos_inertial_x[1][0]])

    drone_body_y.set_xdata([statesTotal_x[num]+drone_pos_inertial_y_neg[0][0],statesTotal_x[num]+drone_pos_inertial_y[0][0]])
    drone_body_y.set_ydata([statesTotal_y[num]+drone_pos_inertial_y_neg[1][0],statesTotal_y[num]+drone_pos_inertial_y[1][0]])

    real_trajectory.set_xdata(statesTotal_x[0:num])
    real_trajectory.set_ydata(statesTotal_y[0:num])
    real_trajectory.set_3d_properties(statesTotal_z[0:num])

    if matplotlib.__version__ == "3.1.3" or matplotlib.__version__ == "3.2.0rc3" or matplotlib.__version__ == "3.2.0" \
    or matplotlib.__version__ == "3.2.1" or matplotlib.__version__ == "3.2.2":
        drone_body_x.set_3d_properties([statesTotal_z[num]+drone_pos_inertial_x_neg[2][0],statesTotal_z[num]+drone_pos_inertial_x[2][0]])
        drone_body_y.set_3d_properties([statesTotal_z[num]+drone_pos_inertial_y_neg[2][0],statesTotal_z[num]+drone_pos_inertial_y[2][0]])

    if sim_version==1:
        drone_body_phi.set_data([-length_y*0.9*0.9,length_y*0.9*0.9],[drone_pos_inertial_y_neg[2][0],drone_pos_inertial_y[2][0]])
        drone_body_theta.set_data([length_x*0.9*0.9,-length_x*0.9*0.9],[drone_pos_inertial_x[2][0],drone_pos_inertial_x_neg[2][0]])
        U1_function.set_data(t_ani[0:num],UTotal_U1[0:num])
        U2_function.set_data(t_ani[0:num],UTotal_U2[0:num])
        U3_function.set_data(t_ani[0:num],UTotal_U3[0:num])
        U4_function.set_data(t_ani[0:num],UTotal_U4[0:num])

        return drone_body_x, drone_body_y, real_trajectory,\
        drone_body_phi, drone_body_theta, U1_function, U2_function, U3_function, U4_function

    else:
        drone_position_x.set_data(t_ani[0:num],statesTotal_x[0:num])
        drone_position_y.set_data(t_ani[0:num],statesTotal_y[0:num])
        drone_position_z.set_data(t_ani[0:num],statesTotal_z[0:num])
        drone_orientation_phi.set_data(t_ani[0:num],statesTotal_phi[0:num])
        drone_orientation_theta.set_data(t_ani[0:num],statesTotal_theta[0:num])
        drone_orientation_psi.set_data(t_ani[0:num],statesTotal_psi[0:num])

        return drone_body_x, drone_body_y, real_trajectory,\
        drone_position_x, drone_position_y, drone_position_z,\
        drone_orientation_phi, drone_orientation_theta, drone_orientation_psi

# Set up figure properties
fig_x = 16
fig_y = 9
n = 4
m = 3
fig = plt.figure(figsize=(fig_x, fig_y),dpi=200,facecolor=(0.9,0.9,0.9))
gs = gridspec.GridSpec(n,m)

# Create an object for the drone
ax0 = fig.add_subplot(gs[0:3,0:2],projection='3d',facecolor=(0.9,0.9,0.9))

# Plot reference trajectory
ref_trajectory = ax0.plot(x_ref,y_ref,z_ref,'b',lw=1,label='reference')
real_trajectory = ax0.plot([],[],[],'r',lw=1,label='trajectory')
drone_body_x = ax0.plot([],[],[],'r',lw=5,label='drone_x')
drone_body_y = ax0.plot([],[],[],'g',lw=5,label='drone_y')

ax0.set_xlim(min_ref, max_ref)
ax0.set_ylim(min_ref, max_ref)
ax0.set_zlim(0, max(z_ref))

ax0.set_xlabel('X [m]')
ax0.set_ylabel('Y [m]')
ax0.set_zlabel('Z [m]')
ax0.legend(loc='upper left')

if matplotlib.__version__ != "3.1.3" and matplotlib.__version__ != "3.2.0rc3" and matplotlib.__version__ != "3.2.0" \
and matplotlib.__version__ != "3.2.1" and matplotlib.__version__ != "3.2.2":
    version_warning=plt.title('For full simulation, Matplotlib 3.2.2 needed! Your version is '+matplotlib.__version__+'! Please refer to Python installation videos.',color='r')

if sim_version==1:

    # VERSION 1

    # Drone orientation (phi - around x axis) - zoomed:
    ax1=fig.add_subplot(gs[3,0],facecolor=(0.9,0.9,0.9))
    drone_body_phi,=ax1.plot([],[],'--g',linewidth=2,label='drone_y (+: Z-up,Y-right,phi-CCW)')
    ax1.set_xlim(-length_y*0.9,length_y*0.9)
    ax1.set_ylim(-length_y*1.1*0.01,length_y*1.1*0.01)
    ax1.legend(loc='upper left',fontsize='small')
    plt.grid(True)

    # Drone orientation (theta - around y axis) - zoomed:
    ax2=fig.add_subplot(gs[3,1],facecolor=(0.9,0.9,0.9))
    drone_body_theta,=ax2.plot([],[],'--r',linewidth=2,label='drone_x (+: Z-up,X-left,theta-CCW)')
    ax2.set_xlim(length_x*0.9,-length_x*0.9)
    ax2.set_ylim(-length_x*1.1*0.01,length_x*1.1*0.01)
    ax2.legend(loc='upper left',fontsize='small')
    plt.grid(True)

    # Create the function for U1
    ax3=fig.add_subplot(gs[0,2],facecolor=(0.9,0.9,0.9))
    U1_function,=ax3.plot([],[],'b',linewidth=1,label='Thrust (U1) [N]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(UTotal_U1)-0.01,np.max(UTotal_U1)+0.01)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    # Create the function for U2
    ax4=fig.add_subplot(gs[1,2],facecolor=(0.9,0.9,0.9))
    U2_function,=ax4.plot([],[],'b',linewidth=1,label='Roll (U2) [Nm]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(UTotal_U2)-0.01,np.max(UTotal_U2)+0.01)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    # Create the function for U3
    ax5=fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
    U3_function,=ax5.plot([],[],'b',linewidth=1,label='Pitch (U3) [Nm]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(UTotal_U3)-0.01,np.max(UTotal_U3)+0.01)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    # Create the function for U4
    ax6=fig.add_subplot(gs[3,2],facecolor=(0.9,0.9,0.9))
    U4_function,=ax6.plot([],[],'b',linewidth=1,label='Yaw (U4) [Nm]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(UTotal_U4)-0.01,np.max(UTotal_U4)+0.01)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')
    plt.xlabel('t-time [s]',fontsize=15)

else:
    # VERSION 2

    # Drone position: X
    ax1=fig.add_subplot(gs[3,0],facecolor=(0.9,0.9,0.9))
    ax1.plot(t,x_ref,'b',linewidth=1,label='X_ref [m]')
    drone_position_x,=ax1.plot([],[],'r',linewidth=1,label='X [m]')
    ax1.set_xlim(0,t_ani[-1])
    ax1.set_ylim(np.min(statesTotal_x)-0.01,np.max(statesTotal_x)+0.01)
    ax1.legend(loc='lower right',fontsize='small')
    plt.grid(True)
    plt.xlabel('t-time [s]',fontsize=15)

    # Drone position: Y
    ax2=fig.add_subplot(gs[3,1],facecolor=(0.9,0.9,0.9))
    ax2.plot(t,y_ref,'b',linewidth=1,label='Y_ref [m]')
    drone_position_y,=ax2.plot([],[],'r',linewidth=1,label='Y [m]')
    ax2.set_xlim(0,t_ani[-1])
    ax2.set_ylim(np.min(statesTotal_y)-0.01,np.max(statesTotal_y)+0.01)
    ax2.legend(loc='lower right',fontsize='small')
    plt.grid(True)
    plt.xlabel('t-time [s]',fontsize=15)

    # Drone position: Z
    ax3=fig.add_subplot(gs[3,2],facecolor=(0.9,0.9,0.9))
    ax3.plot(t,z_ref,'b',linewidth=1,label='Z_ref [m]')
    drone_position_z,=ax3.plot([],[],'r',linewidth=1,label='Z [m]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(statesTotal_z)-0.01,np.max(statesTotal_z)+0.01)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize='small')
    plt.xlabel('t-time [s]',fontsize=15)

    # Create the function for Phi
    ax4=fig.add_subplot(gs[0,2],facecolor=(0.9,0.9,0.9))
    ax4.plot(t_angles,ref_angles_total[:,0],'b',linewidth=1,label='Phi_ref [rad]')
    drone_orientation_phi,=ax4.plot([],[],'r',linewidth=1,label='Phi [rad]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(statesTotal_phi)-0.01,np.max(statesTotal_phi)+0.01)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize='small')

    # Create the function for Theta
    ax5=fig.add_subplot(gs[1,2],facecolor=(0.9,0.9,0.9))
    ax5.plot(t_angles,ref_angles_total[:,1],'b',linewidth=1,label='Theta_ref [rad]')
    drone_orientation_theta,=ax5.plot([],[],'r',linewidth=1,label='Theta [rad]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(statesTotal_theta)-0.01,np.max(statesTotal_theta)+0.01)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize='small')

    # Create the function for Psi
    ax6=fig.add_subplot(gs[2,2],facecolor=(0.9,0.9,0.9))
    ax6.plot(t_angles,ref_angles_total[:,2],'b',linewidth=1,label='Psi_ref [rad]')
    drone_orientation_psi,=ax6.plot([],[],'r',linewidth=1,label='Psi [rad]')
    plt.xlim(0,t_ani[-1])
    plt.ylim(np.min(statesTotal_psi)-0.01,np.max(statesTotal_psi)+0.01)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize='small')

drone_ani=animation.FuncAnimation(fig, update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)
plt.show()

no_plots=support.constants['no_plots']
if no_plots==1:
    exit()

else:
    # Plot the world
    ax=plt.axes(projection='3d')
    ax.plot(x_ref,y_ref,z_ref,'b',label='reference')
    ax.plot(statesTotal_x,statesTotal_y,statesTotal_z,'r',label='trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    copyright=ax.text(0,max(y_ref),max(z_ref)*1.20,'© Mark Misin Engineering',size=15)
    ax.legend()
    plt.show()


    # Position and velocity plots
    plt.subplot(2,1,1)
    plt.plot(t,x_ref,'b',linewidth=1,label='X_ref')
    plt.plot(t_ani,statesTotal_x,'r',linewidth=1,label='X')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('X [m]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right',fontsize='small')

    plt.subplot(2,1,2)
    plt.plot(t,x_dot_ref,'b',linewidth=1,label='X_dot_ref')
    plt.plot(t,velocityXYZ_total[0:len(velocityXYZ_total):innerDyn_length,0],'r',linewidth=1,label='X_dot')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('X_dot [m/s]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right',fontsize='small')
    plt.show()

    # print(Y_dot_ref[0]-velocityXYZ_total[0,1])
    plt.subplot(2,1,1)
    plt.plot(t,y_ref,'b',linewidth=1,label='Y_ref')
    plt.plot(t_ani,statesTotal_y,'r',linewidth=1,label='Y')
    # plt.plot(t,Y_ref-statesTotal2[:,7],'g',linewidth=1,label='D')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('Y [m]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right',fontsize='small')

    plt.subplot(2,1,2)
    plt.plot(t,y_dot_ref,'b',linewidth=1,label='Y_dot_ref')
    plt.plot(t,velocityXYZ_total[0:len(velocityXYZ_total):innerDyn_length,1],'r',linewidth=1,label='Y_dot')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('Y_dot [m/s]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right',fontsize='small')
    plt.show()


    plt.subplot(2,1,1)
    plt.plot(t,z_ref,'b',linewidth=1,label='Z_ref')
    plt.plot(t_ani,statesTotal_z,'r',linewidth=1,label='Z')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('Z [m]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right',fontsize='small')

    plt.subplot(2,1,2)
    plt.plot(t,z_dot_ref,'b',linewidth=1,label='Z_dot_ref')
    plt.plot(t,velocityXYZ_total[0:len(velocityXYZ_total):innerDyn_length,2],'r',linewidth=1,label='Z_dot')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('Z_dot [m/s]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='center right',fontsize='small')
    plt.show()


    # Orientation plots
    plt.subplot(3,1,1)
    plt.plot(t_angles,ref_angles_total[:,0],'b',linewidth=1,label='Phi_ref')
    plt.plot(t_ani,statesTotal_phi,'r',linewidth=1,label='Phi')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('Phi [rad]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize='small')

    plt.subplot(3,1,2)
    plt.plot(t_angles,ref_angles_total[:,1],'b',linewidth=1,label='Theta_ref')
    plt.plot(t_ani,statesTotal_theta,'r',linewidth=1,label='Theta')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('Theta [rad]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize='small')

    plt.subplot(3,1,3)
    plt.plot(t_angles,ref_angles_total[:,2],'b',linewidth=1,label='Psi_ref')
    plt.plot(t_ani,statesTotal_psi,'r',linewidth=1,label='Psi')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('Psi [rad]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='lower right',fontsize='small')
    plt.show()


    # Control input plots
    plt.subplot(4,2,1)
    plt.plot(t_angles,UTotal[0:len(UTotal),0],'b',linewidth=1,label='U1')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('U1 [N]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    plt.subplot(4,2,3)
    plt.plot(t_angles,UTotal[0:len(UTotal),1],'b',linewidth=1,label='U2')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('U2 [Nm]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    plt.subplot(4,2,5)
    plt.plot(t_angles,UTotal[0:len(UTotal),2],'b',linewidth=1,label='U3')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('U3 [Nm]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    plt.subplot(4,2,7)
    plt.plot(t_angles,UTotal[0:len(UTotal),3],'b',linewidth=1,label='U4')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('U4 [Nm]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    plt.subplot(4,2,2)
    plt.plot(t_angles,omegas_bundle[0:len(omegas_bundle),0],'b',linewidth=1,label='omega 1')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('omega 1 [rad/s]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    plt.subplot(4,2,4)
    plt.plot(t_angles,omegas_bundle[0:len(omegas_bundle),1],'b',linewidth=1,label='omega 2')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('omega 2 [rad/s]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    plt.subplot(4,2,6)
    plt.plot(t_angles,omegas_bundle[0:len(omegas_bundle),2],'b',linewidth=1,label='omega 3')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('omega 3 [rad/s]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')

    plt.subplot(4,2,8)
    plt.plot(t_angles,omegas_bundle[0:len(omegas_bundle),3],'b',linewidth=1,label='omega 4')
    plt.xlabel('t-time [s]',fontsize=15)
    plt.ylabel('omega 4 [rad/s]',fontsize=15)
    plt.grid(True)
    plt.legend(loc='upper right',fontsize='small')
    plt.show()