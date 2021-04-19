import numpy as np
import matplotlib.pyplot as plt
from rotation_matrices import RotationMatrix

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

    def pos_controller(self, x_ref, x_dot_ref, x_dot_dot_ref, y, y_dot_ref, y_dot_dot_ref, z, z_dot_ref, z_dot_dot_ref, psi_ref, states):
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



