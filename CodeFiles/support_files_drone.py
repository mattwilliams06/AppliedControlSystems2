import numpy as np
import matplotlib.pyplot as plt
from rotation_matrices import RotationMatrix
import scipy.linalg

class SupportFilesDrone:

    def __init__(self):
        self.constants = {}
        # Constants (Astec Hummingbird)
        Ix = 0.0034 # kg*m^2
        Iy = 0.0034 # kg*m^2
        Iz  = 0.006 # kg*m^2
        m  = 0.698 # kg
        g  = 9.81 # m/s^2
        Jtp=1.302*10**(-6) # N*m*s^2=kg*m^2, propellor moment of inertia
        Ts=0.1 # s
        self.constants['Ix'] = Ix
        self.constants['Iy'] = Iy
        self.constants['Iz'] = Iz
        self.constants['m'] = m
        self.constants['g'] = g
        self.constants['Jtp'] = Jtp
        self.constants['Ts'] = Ts

        # Matrix weights for the cost function (They must be diagonal)
        Q=np.diag([10,10,10],k=0) # weights for outputs (all samples, except the last one)
        S=np.diag([20,20,20],k=0) # weights for the final horizon period outputs
        R=np.diag([10,10,10],k=0) # weights for inputs
        self.constants['Q'] = Q
        self.constants['S'] = S
        self.constants['R'] = R

        ct = 7.6184*10**(-8)*(60/(2*np.pi))**2 # thrust coefficient [N-s^2]
        cq = 2.6839*10**(-9)*(60/(2*np.pi))**2 # torque coefficient [N-m-s^2]
        l = 0.171 # distance from drone COG to motor COG [m]
        self.constants['ct'] = ct
        self.constants['cq'] = cq
        self.constants['l'] = l

        controlled_states=3 # Number of attitude outputs: Phi, Theta, Psi
        hz = 4 # horizon period
        self.constants['controlled_states'] = controlled_states
        self.constants['hz'] = hz

        innerDyn_length=4 # Number of inner control loop iterations
        self.constants['innerDyn_length'] = innerDyn_length

        # The poles
        px=np.array([-1,-2])
        py=np.array([-1,-2])
        pz=np.array([-1,-2])
        self.constants['px'] = px
        self.constants['py'] = py
        self.constants['pz'] = pz

        # # Complex poles
        # px=np.array([-0.1+0.3j,-0.1-0.3j])
        # py=np.array([-0.1+0.3j,-0.1-0.3j])
        # pz=np.array([-1+1.3j,-1-1.3j])

        # trajectory values
        r=2             # trajectory radius
        f=0.025         # trajectory frequency
        height_i=5      # initial trajectory height
        height_f=25     # final trajectory height
        self.constants['r'] = r
        self.constants['f'] = f
        self.constants['height_i'] = height_i
        self.constants['height_f'] = height_f

        pos_x_y=0 # Default: 0. Make positive x and y longer for visual purposes (1-Yes, 0-No). It does not affect the dynamics of the UAV.
        sub_loop=5 # for animation purposes
        sim_version=1 # Can only be 1 or 2 - depending on that, it will show you different graphs in the animation
        self.constants['pos_x_y'] = 0
        self.constants['sub_loop'] = 5
        self.constants['sim_version'] = sim_version

        # Drag force:
        drag_switch=0 # Must be either 0 or 1 (0 - drag force OFF, 1 - drag force ON)
        self.constants['drag_switch'] = drag_switch

        # Drag force coefficients [-]:
        C_D_u=1.5
        C_D_v=1.5
        C_D_w=2.0
        self.constants['C_D_u'] = C_D_u
        self.constants['C_D_v'] = C_D_v
        self.constants['C_D_w'] = C_D_w

        # Drag force cross-section area [m^2]
        A_u=2*l*0.01+0.05**2
        A_v=2*l*0.01+0.05**2
        A_w=2*2*l*0.01+0.05**2
        self.constants['A_u'] = A_u
        self.constants['A_v'] = A_v
        self.constants['A_w'] = A_w

        # Air density
        rho=1.225 # [kg/m^3]
        trajectory=8 # Choose the trajectory: only from 1-9
        no_plots=0 # 0-you will see the plots; 1-you will skip the plots (only animation)
        self.constants['rho'] = rho
        self.constants['trajectory'] = trajectory
        self.constants['no_plots'] = no_plots

        return None

    def trajectory_generator(self, t):

        Ts=self.constants['Ts']
        innerDyn_length=self.constants['innerDyn_length']
        r=self.constants['r']
        f=self.constants['f']
        height_i=self.constants['height_i']
        height_f=self.constants['height_f']
        trajectory=self.constants['trajectory']
        d_height=height_f-height_i

        # Define the x, y, z dimensions for the drone trajectories
        alpha=2*np.pi*f*t   # current angle

        if trajectory==1 or trajectory==2 or trajectory==3 or trajectory==4:
            # Writing the logic statements this way allows us to keep the same x, y, and z
            # Trajectory 1
            x=r*np.cos(alpha)   # x-position, alpha(t) = 2*pi*f*t
            y=r*np.sin(alpha)   # y-position
            z=height_i+d_height/(t[-1])*t # z-position as a function of time

            x_dot=-r*np.sin(alpha)*2*np.pi*f    # x-velocity for counter-clockwise circular motion
            y_dot=r*np.cos(alpha)*2*np.pi*f     # t-velocity for counter-clockwise circular motion
            z_dot=d_height/(t[-1])*np.ones(len(t))

            x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2 
            y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
            z_dot_dot=0*np.ones(len(t))

            if trajectory==2:
                # Trajectory 2
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+d_height/t[-1]*(t[101:len(t)]-t[100])

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot*(t/20))]=d_height/(t[-1])*np.ones(len(t[101:len(t)]))

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=0*np.ones(len(t[101:len(t)]))

            elif trajectory==3:
                # Trajectory 3
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+d_height/t[-1]*(t[101:len(t)]-t[100])**2

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot)]=2*d_height/(t[-1])*(t[101:len(t)]-t[100])

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=2*d_height/t[-1]*np.ones(len(t[101:len(t)]))

            elif trajectory==4:
                # Trajectory 4
                # Make sure you comment everything except Trajectory 1 and this bonus trajectory
                x[101:len(x)]=2*(t[101:len(t)]-t[100])/20+x[100]
                y[101:len(y)]=2*(t[101:len(t)]-t[100])/20+y[100]
                z[101:len(z)]=z[100]+50*d_height/t[-1]*np.sin(0.1*(t[101:len(t)]-t[100]))

                x_dot[101:len(x_dot)]=1/10*np.ones(len(t[101:len(t)]))
                y_dot[101:len(y_dot)]=1/10*np.ones(len(t[101:len(t)]))
                z_dot[101:len(z_dot)]=5*d_height/t[-1]*np.cos(0.1*(t[101:len(t)]-t[100]))

                x_dot_dot[101:len(x_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                y_dot_dot[101:len(y_dot_dot)]=0*np.ones(len(t[101:len(t)]))
                z_dot_dot[101:len(z_dot_dot)]=-0.5*d_height/t[-1]*np.sin(0.1*(t[101:len(t)]-t[100]))

        elif trajectory==5 or trajectory==6:
            if trajectory==5:
                power=1
            else:
                power=2

            if power == 1:
                # Trajectory 5
                r_2=r/15
                x=(r_2*t**power+r)*np.cos(alpha)
                y=(r_2*t**power+r)*np.sin(alpha)
                z=height_i+d_height/t[-1]*t

                x_dot=r_2*np.cos(alpha)-(r_2*t+r)*np.sin(alpha)*2*np.pi*f
                y_dot=r_2*np.sin(alpha)+(r_2*t+r)*np.cos(alpha)*2*np.pi*f
                z_dot=d_height/(t[-1])*np.ones(len(t))

                x_dot_dot=-r_2*np.sin(alpha)*4*np.pi*f-(r_2*t+r)*np.cos(alpha)*(2*np.pi*f)**2
                y_dot_dot=r_2*np.cos(alpha)*4*np.pi*f-(r_2*t+r)*np.sin(alpha)*(2*np.pi*f)**2
                z_dot_dot=0*np.ones(len(t))
            else:
                # Trajectory 6
                r_2=r/500
                x=(r_2*t**power+r)*np.cos(alpha)
                y=(r_2*t**power+r)*np.sin(alpha)
                z=height_i+d_height/t[-1]*t

                x_dot=power*r_2*t**(power-1)*np.cos(alpha)-(r_2*t**(power)+r)*np.sin(alpha)*2*np.pi*f
                y_dot=power*r_2*t**(power-1)*np.sin(alpha)+(r_2*t**(power)+r)*np.cos(alpha)*2*np.pi*f
                z_dot=d_height/(t[-1])*np.ones(len(t))

                x_dot_dot=(power*(power-1)*r_2*t**(power-2)*np.cos(alpha)-power*r_2*t**(power-1)*np.sin(alpha)*2*np.pi*f)-(power*r_2*t**(power-1)*np.sin(alpha)*2*np.pi*f+(r_2*t**power+r_2)*np.cos(alpha)*(2*np.pi*f)**2)
                y_dot_dot=(power*(power-1)*r_2*t**(power-2)*np.sin(alpha)+power*r_2*t**(power-1)*np.cos(alpha)*2*np.pi*f)+(power*r_2*t**(power-1)*np.cos(alpha)*2*np.pi*f-(r_2*t**power+r_2)*np.sin(alpha)*(2*np.pi*f)**2)
                z_dot_dot=0*np.ones(len(t))

        elif trajectory==7:
        # Trajectory 7
            x=2*t/20+1
            y=2*t/20-2
            z=height_i+d_height/t[-1]*t

            x_dot=1/10*np.ones(len(t))
            y_dot=1/10*np.ones(len(t))
            z_dot=d_height/(t[-1])*np.ones(len(t))

            x_dot_dot=0*np.ones(len(t))
            y_dot_dot=0*np.ones(len(t))
            z_dot_dot=0*np.ones(len(t))

        elif trajectory==8:
            # Trajectory 8
            x=r/5*np.sin(alpha)+t/100
            y=t/100-1
            z=height_i+d_height/t[-1]*t

            x_dot=r/5*np.cos(alpha)*2*np.pi*f+1/100
            y_dot=1/100*np.ones(len(t))
            z_dot=d_height/(t[-1])*np.ones(len(t))

            x_dot_dot=-r/5*np.sin(alpha)*(2*np.pi*f)**2
            y_dot_dot=0*np.ones(len(t))
            z_dot_dot=0*np.ones(len(t))

        elif trajectory==9:
            # Trajectory 9
            wave_w=1
            x=r*np.cos(alpha)
            y=r*np.sin(alpha)
            z=height_i+7*d_height/t[-1]*np.sin(wave_w*t)

            x_dot=-r*np.sin(alpha)*2*np.pi*f
            y_dot=r*np.cos(alpha)*2*np.pi*f
            z_dot=7*d_height/(t[-1])*np.cos(wave_w*t)*wave_w

            x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
            y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
            z_dot_dot=-7*d_height/(t[-1])*np.sin(wave_w*t)*wave_w**2

        else:
            print('There are only 9 trajectories. Please select 1-9.')
            exit()

        dx = x[1:] - x[:-1]
        dy = y[1:] - x[:-1]
        dz = z[1:] - z[:-1]
        dx = np.append(np.array(dx[0]), dx)
        dy = np.append(np.array(dy[0]), dy)
        dz = np.append(np.array(dz[0]), dz)

        # reference yaw angles
        psi = np.zeros_like(x)
        psiInt = np.zeros_like(psi)
        psi = np.arctan2(y,x)
        dpsi = psi[1:] - psi[:-1]

        for i in range(1,len(psiInt)):
            if dpsi[i-1] < -np.pi:
                psiInt[i] = psiInt[i-1] + dpsi[i-1] + 2*np.pi
            elif dpsi[i-1] > np.pi:
                psiInt[i] = psiInt[i-1] + dpsi[i-1] - 2*np.pi
            else:
                psiInt[i] = psiInt[i-1] + dpsi[i-1]

        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, psiInt

    def pos_controller(self, x_ref, x_dot_ref, x_dot_dot_ref, y_ref, y_dot_ref, y_dot_dot_ref, z_ref, z_dot_ref, z_dot_dot_ref, psi_ref, states):
        ''' Position controller -- computes the U1 for the open-loop system, and the phi and theta angles for the MPC controller'''
        # load constants
        m = self.constants['m']
        g = self.constants['g']
        px = self.constants['px']  # in the form [lambda1, lambda2]
        py = self.constants['py']
        pz = self.constants['pz']

        # create the state states (u,v,w,p,q,r,x,y,z,phi,theta,psi)
        u = states[0]
        v = states[1]
        w = states[2]
        x = states[6]
        y = states[7]
        z = states[8]
        phi = states[9]
        theta = states[10]
        psi = states[11]

        # instantiate rotation matrix
        R_matrix = RotationMatrix(['x','y','z'])
        pos_vel_body = np.array([u,v,w])
        pos_vel_fixed = R_matrix([phi, theta, psi], pos_vel_body)
        x_dot = pos_vel_fixed[0]
        y_dot = pos_vel_fixed[1]
        z_dot = pos_vel_fixed[2]

        # compute the errors
        ex = x_ref - x
        ex_dot = x_dot_ref - x_dot
        ey = y_ref - y
        ey_dot = y_dot_ref - y_dot
        ez = z_ref - z
        ez_dot = z_dot_ref - z_dot

        # compute the error gain terms
        k1x = (px[0] - (px[0]+px[1])/2)**2 - (px[0]+px[1])**2/4
        k2x = px[0] + px[1]
        k1x = k1x.real
        k2x = k2x.real

        k1y = (py[0] - (py[0]+py[1])/2)**2 - (py[0]+py[1])**2/4
        k2y = py[0] + py[1]
        k1y = k1y.real
        k2y = k2y.real

        k1z = (pz[0] - (pz[0]+pz[1])/2)**2 - (pz[0]+pz[1])**2/4
        k2z = pz[0] + pz[1]
        k1z = k1z.real
        k2z = k2z.real

        # Compute the values vx, vy, vz for the position controller
        ux = ex*k1x + ex_dot*k2x
        uy = ey*k1y + ey_dot*k2y
        uz = ez*k1z + ez_dot*k2z

        ######### ARE UX, UY, AND UZ LISTS LIKE WHAT MARK'S CODE SUGGESTS???? ##########
        vx = x_dot_dot_ref - ux # this is really x_dot_dot
        vy = y_dot_dot_ref - uy
        vz = z_dot_dot_ref - uz

        # compute phi, theta, and U1
        a = vx/(vz + g)
        b = vy/(vz + g)
        c = np.cos(psi_ref)
        d = np.sin(psi_ref)

        tan_theta = a*c + b*d
        theta_ref = np.arctan(tan_theta)

        # Make psi_ref be between -2pi and 2pi
        if psi_ref >= 0:
            psi_ref_singularity = psi_ref - np.floor(psi_ref/(2*np.pi))*2*np.pi
        else:
            psi_ref_singularity = psi_ref + np.floor(psi_ref/(2*np.pi))*2*np.pi

        # Recall that we want to choose the equation for tan_phi so that we do not divide by zero.
        # The equation for tan_phi will be determined by which quadrant phi_ref is in
        if (psi_ref_singularity < np.pi/4) or (psi_ref_singularity) > 7*np.pi/4 or (psi_ref_singularity > 3*np.pi/4 and psi_ref_singularity < 5*np.pi/4):
            tan_phi = (np.cos(theta_ref)*(tan_theta*d-b))/c
        else:
            tan_phi = (np.cos(theta_ref)*(a - tan_theta*c))/d
        
        phi_ref = np.arctan(tan_phi)
        U1 = m*(vz+g)/(np.cos(phi_ref)*np.cos(theta_ref))

        return phi_ref, theta_ref, U1

    def LPV_cont_discrete(self, states, omega_total):
        ''' This is an LPV model concerning the three rotational axes '''

        # Get the necessary constants
        Ix = self.constants['Ix']
        Iy = self.constants['Iy']
        Iz = self.constants['Iz']
        Jtp = self.constants['Jtp']
        Ts = self.constants['Ts']

        # Assign the states
        # States: [u,v,w,p,q,r,x,y,z,phi,theta,psi]
        u = states[0]
        v = states[1]
        w = states[2]
        p = states[3]
        q = states[4]
        r = states[5]
        phi = states[9]
        theta = states[10]
        psi = states[11]

        # rotational matrix that relates u,v,w with x_dot, y_dot, z_dot
        Rmat = RotationMatrix(['x','y','z'])
        pos_vel_body = np.array([u,v,w])
        pos_vel_fixed = Rmat([phi,theta,psi],pos_vel_body)
        x_dot = pos_vel_fixed[0]
        y_dot = pos_vel_fixed[1]
        z_dot = pos_vel_fixed[2]
        # I only think the code below is necessary in Mark's code, where he uses a column vector in the matrix
        # multiplication, and therefore gets a 2D array as output
        # x_dot = x_dot[0]
        # y_dot = y_dot[0]
        # z_dot = z_dot[0]

        # To get phi_dot, theta_dot, and psi_dot, we need the transformation matrix
        # This is the matrix that relates the body frame rotation rates p, q, and r to 
        # phi_dot, theta_dot, and psi_dot
        Rx = Rmat._Rx(phi)
        Ry = Rmat._Ry(theta)
        mat_coef = []
        for i in range(3):
            mat = np.zeros((3,3))
            mat[i,i] = 1
            mat_coef.append(mat)
        Tinv = mat_coef[0] + np.linalg.inv(Rx)@mat_coef[1] + np.linalg.inv(Ry@Rx)@mat_coef[2]
        Tmat = np.linalg.inv(Tinv)
        rot_vel_body = np.array([p,q,r])
        rot_vel_fixed = Tmat@rot_vel_body
        phi_dot = rot_vel_fixed[0]
        theta_dot = rot_vel_fixed[1]
        psi_dot = rot_vel_fixed[2]

        # Create the continuous LPV A, B, C, D matrices
        A01 = 1
        A13 = Jtp*omega_total/Ix
        A15 = theta_dot*(Iy-Iz)/Ix
        A23 = 1
        A31 = Jtp*omega_total/Iy
        A35 = phi_dot*(Iz-Ix)/Iy
        A45 = 1
        A51 = theta_dot/2*(Ix-Iy)/Iz
        A53 = theta_dot/2*(Ix-Iy)/Iz

        A = np.zeros((6,6))
        B = np.zeros((6,3))
        C = np.zeros((3,6))
        D = np.zeros((3,3))

        A[0,1] = A01
        A[1,3] = A13
        A[1,5] = A15
        A[2,3] = A23
        A[3,1] = A31
        A[3,5] = A35
        A[4,5] = A45
        A[5,1] = A51
        A[5,3] = A53

        B[1,0] = 1/Ix
        B[3,1] = 1/Iy
        B[5,2] = 1/Iz

        C[0,0] = 1
        C[1,2] = 1
        C[2,4] = 1

        # Discretize the system using forward Euler
        Ad = np.eye(np.size(A,1)) + Ts*A
        Bd = Ts*B
        Cd = C
        Dd = D

        return Ad, Bd, Cd, Dd, x_dot, y_dot, z_dot, phi, phi_dot, theta, theta_dot, psi, psi_dot

    def mpc_simplification(self, Ad, Bd, Cd, Dd, hz):
        ''' Create the larger matrices for MPC simplification '''
        # big_zero = np.zeros((np.size(Bd,1), np.size(Ad,1)))
        # big_I = np.eye(np.size(Bd,1))
        # A_aug = np.block([[Ad, Bd],[big_zero, big_I]])
        # B_aug = np.block([[Bd],[big_I]])
        # big_zero = np.zeros((np.size(Cd,0),np.size(Bd,1)))
        # C_aug = np.block([Cd, big_zero])
        # print('A aug shape: ', A_aug.shape)
        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug=Dd

        Q = self.constants['Q']
        S = self.constants['S']
        R = self.constants['R']

        CQC = C_aug.T@Q@C_aug
        CSC = C_aug.T@S@C_aug
        QC = Q@C_aug
        SC = S@C_aug

        CQC_list = [CQC]*(hz-1)
        # Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Qdb = scipy.linalg.block_diag(*CQC_list, CSC)
        QC_list = [QC]*(hz-1)
        Tdb = scipy.linalg.block_diag(*QC_list, SC)
        R_list = [R]*hz
        Rdb = scipy.linalg.block_diag(*R_list)
        A2 = np.linalg.matrix_power(A_aug, 2)
        A3 = np.linalg.matrix_power(A_aug, 3)
        A4 = np.linalg.matrix_power(A_aug, 4)
        B_zeros = np.zeros_like(B_aug)
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for row in range(hz):
            for col in range(hz):
                if row <= col:
                    Cdb[B_aug.shape[0]*row:B_aug.shape[0]*row+B_aug.shape[0],B_aug.shape[1]*col:B_aug.shape[1]*col+B_aug.shape[1]] = \
                        np.matmul(np.linalg.matrix_power(A_aug,((row+1)-(col+1))),B_aug)
                    
        # Cdb = np.block([[B_aug, B_zeros, B_zeros, B_zeros],
        #                 [A2@B_aug, B_aug, B_zeros, B_zeros],
        #                 [A3@B_aug, A2@B_aug, B_aug, B_zeros],
        #                 [A4@B_aug, A3@B_aug, A2@B_aug, B_aug]])
            Adc[np.size(A_aug,0)*row:np.size(A_aug,0)*row+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,row+1)
        # Adc = np.block([[A_aug],[A2],[A3],[A4]])

        # print('Cdb.T shape: ', Cdb.T.shape)
        # print('Qdb shape: ', Qdb.shape)
        # print('Cdb shape: ', Cdb.shape)
        # print('Rdb shape: ', Rdb.shape)
        # print('B aug: ', B_aug.shape)
        Hdb = Cdb.T@Qdb@Cdb + Rdb
        Fdbt = np.block([[Adc.T@Qdb@Cdb],[-Tdb@Cdb]])

        return Hdb, Fdbt, Cdb, Adc
        # '''This function creates the compact matrices for Model Predictive Control'''
        # # db - double bar
        # # dbt - double bar transpose
        # # dc - double circumflex
        # A_aug=np.concatenate((Ad,Bd),axis=1)
        # temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        # temp2=np.identity(np.size(Bd,1))
        # temp=np.concatenate((temp1,temp2),axis=1)

        # A_aug=np.concatenate((A_aug,temp),axis=0)
        # B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        # C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        # D_aug=Dd


        # # Q=self.constants[7]
        # # S=self.constants[8]
        # # R=self.constants[9]

        # CQC=np.matmul(np.transpose(C_aug),Q)
        # CQC=np.matmul(CQC,C_aug)

        # CSC=np.matmul(np.transpose(C_aug),S)
        # CSC=np.matmul(CSC,C_aug)

        # QC=np.matmul(Q,C_aug)
        # SC=np.matmul(S,C_aug)


        # Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        # Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        # Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        # Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        # Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        # for i in range(0,hz):
        #     if i == hz-1:
        #         Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
        #         Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
        #     else:
        #         Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
        #         Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

        #     Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

        #     for j in range(0,hz):
        #         if j<=i:
        #             Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

        #     Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        # Hdb=np.matmul(np.transpose(Cdb),Qdb)
        # Hdb=np.matmul(Hdb,Cdb)+Rdb

        # temp=np.matmul(np.transpose(Adc),Qdb)
        # temp=np.matmul(temp,Cdb)

        # temp2=np.matmul(-Tdb,Cdb)
        # Fdbt=np.concatenate((temp,temp2),axis=0)
        # print('Qdb shape: ', Qdb.shape)
        # print('Cdb shape: ', Cdb.shape)
        # print('Rdb shape: ', Rdb.shape)

        # return Hdb,Fdbt,Cdb,Adc

    def state_space_derivs(self, states, U1, U2, U3, U4, omega_total):
        ''' Need to re-write the return so that it gives the derivatives of the state vector. Need to figure out
        where the angle derivatives come from. '''
        # States: [u,v,w,p,q,r,x,y,z,phi,theta,psi]
        
        Ix = self.constants['Ix']
        Iy = self.constants['Iy']
        Iz = self.constants['Iz']
        m = self.constants['m']
        g = self.constants['g']
        Jtp = self.constants['Jtp']
        u = states[0]
        v = states[1]
        w = states[2]
        p = states[3]
        q = states[4]
        r = states[5]
        x = states[6]
        y = states[7]
        z = states[8]
        phi = states[9]
        theta = states[10]
        psi = states[11]

        # Drag force
        drag_switch = self.constants['drag_switch']
        C_D_u = self.constants['C_D_u']
        C_D_V = self.constants['C_D_v']
        C_D_w = self.constants['C_D_w']
        A_u = self.constants['A_u']
        A_v = self.constants['A_v']
        A_w = self.constants['A_w']
        rho = self.constants['rho']

        if drag_switch==1:
            Fd_u=0.5*C_D_u*rho*u**2*A_u
            Fd_v=0.5*C_D_v*rho*v**2*A_v
            Fd_w=0.5*C_D_w*rho*w**2*A_w
        elif drag_switch==0:
            Fd_u=0
            Fd_v=0
            Fd_w=0
        else:
            print('drag_switch variable must be either 0 or 1 in the __init__ method.')
            exit()

        # Calculate accelerations
        u_dot = (v*r-w*q) + g*np.sin(theta) - Fd_u/m
        v_dot = (w*p-u*r) - g*np.cos(theta)*np.sin(phi) - Fd_v/m
        w_dot = (u*q-v*p) - g*np.cos(theta)*np.cos(phi) + U1/m - Fd_w/m
        p_dot = q*r*(Iy-Iz)/Ix + Jtp/Ix*q*omega_total + U2/Ix
        q_dot = p*r*(Iz-Ix)/Iy - Jtp/Iy*p*omega_total + U3/Iy
        r_dot = p*q*(Ix-Iy)/Iz + U4/Iz

        # Return accelerations
        return np.array([u_dot, v_dot, w_dot, p_dot, q_dot, r_dot])
        
    def open_loop_new_states(self, states, omega_total, U1, U2, U3, U4):
        ''' This methof computed the new state vector one sample time later '''

        # Get the necessary constants
        Ix = self.constants['Ix']
        Iy = self.constants['Iy']
        Iz = self.constants['Iz']
        m = self.constants['m']
        g = self.constants['g']
        Jtp = self.constants['Jtp']
        Ts = self.constants['Ts'] 

        # States: [u,v,w,p,q,r,x,y,z,phi,theta,psi]
        u0 = states[0]
        v0 = states[1]
        w0 = states[2]
        p0 = states[3]
        q0 = states[4]
        r0 = states[5]
        x0 = states[6]
        y0 = states[7]
        z0 = states[8]
        phi0 = states[9]
        theta0 = states[10]
        psi0 = states[11]
        sub_loop = self.constants['sub_loop']
        states_ani = np.zeros((sub_loop,6))
        U_ani = np.zeros((sub_loop,4))

        # Get positon and angular velocities in the inertial frame from the body frame
        Rmat = RotationMatrix(['x','y','z'])
        pos_vel_body = np.array([u0, v0, w0])
        angles = [phi0, theta0, psi0]
        pos_vel_fixed = Rmat(angles, pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]
        rot_vel_body = np.array([p0, q0, r0])
        # Transfer matrix for angular velocities
        T_matrix=np.array([[1,np.sin(phi0)*np.tan(theta0),np.cos(phi0)*np.tan(theta0)],\
                [0,np.cos(phi0),-np.sin(phi0)],\
                [0,np.sin(phi0)/np.cos(theta0),np.cos(phi0)/np.cos(theta0)]])
        rot_vel_body=np.array([p0,q0,r0])
        rot_vel_fixed=np.matmul(T_matrix,rot_vel_body)
        phi_dot=rot_vel_fixed[0]
        theta_dot=rot_vel_fixed[1]
        psi_dot=rot_vel_fixed[2]

        ###### BEGIN RUNGE KUTTA ALGORITHM ######
        accels1 = self.state_space_derivs(states, U1, U2, U3, U4, omega_total)
        k1 = np.concatenate((accels1, [x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]))
        accels2 = self.state_space_derivs(states+k1*Ts/2, U1, U2, U4, U4, omega_total)
        k2 = np.concatenate((accels2, [x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]))
        accels3 = self.state_space_derivs(states+k2*Ts/2, U1, U2, U4, U4, omega_total)
        k3 = np.concatenate((accels3, [x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]))
        accels4 = self.state_space_derivs(states+k3*Ts, U1, U2, U4, U4, omega_total)
        k4 = np.concatenate((accels4, [x_dot, y_dot, z_dot, phi_dot, theta_dot, psi_dot]))

        k = 1/6*(k1 + 2*k2 + 2*k3 + k4)
        new_states = states + k*Ts
        # Do i need the variable declarations below??
        u = new_states[0]
        v = new_states[1]
        w = new_states[2]
        p = new_states[3]
        q = new_states[4]
        r = new_states[5]
        x = new_states[6]
        y = new_states[7]
        z = new_states[8]
        phi = new_states[9]
        theta = new_states[10]
        psi = new_states[11]
        ###### END RUTTA KUTTA ALGORITHM ######
        for k in range(0,sub_loop):
            states_ani[k,0]=x0+(x-x0)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,1]=y0+(y-y0)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,2]=z0+(z-z0)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,3]=phi0+(phi-phi0)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,4]=theta0+(theta-theta0)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,5]=psi0+(psi-psi0)/Ts*(k/(sub_loop-1))*Ts

        U_ani[:,0]=U1
        U_ani[:,1]=U2
        U_ani[:,2]=U3
        U_ani[:,3]=U4

        return new_states, states_ani, U_ani