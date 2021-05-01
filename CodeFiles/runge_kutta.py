import numpy as np
import matplotlib.pyplot as plt

class RK4:

    def __init__(self, derivs, params):
        self.derivs = derivs
        self.params = params

    def _integrate(self, vec, t):
        dt = self.params['dt']
        k1 = self.derivs(vec, t, params)
        k2 = self.derivs(vec+k1*dt/2, t+dt/2, params)
        k3 = self.derivs(vec+k2*dt/2, t+dt/2, params)
        k4 = self.derivs(vec+k3*dt, t+dt, params)

        k = (k1 + 2*k2 + 2*k3 + k4)/6

        return np.array([vec + k*dt])

    def __call__(self, vec, t):
        result = self._integrate(vec, t)
        return result

if __name__ == '__main__':
    # v0 = 965
    # theta = np.pi/4
    # r = 0.00381        # radius of projectile
    # A = np.pi*r**2  # area of projectile
    # vector = np.array([[0,1.6,v0*np.cos(theta), v0*np.sin(theta)]])
    # def derivs(vec, t, params, drag=True):
    #     x, y, vx, vy = vec
    #     g = params['g']
    #     Cd = params['Cd']
    #     A = params['A']
    #     rho = params['rho']
    #     m = params['m']
    #     D = 0.5*rho*Cd*A
    #     # Line below includes drag
    #     if drag:
    #         return np.array([vx, vy, D/m*vx**2*np.sign(vx)*-1, D/m*vy**2*np.sign(vy)*-1-g])
    #     # Line below has no drag
    #     else:
    #         return np.array([vx, vy, 0, -g])
    # t = np.array([[0]])
    # params = {'dt':0.01,'g':9.81,'Cd':0.04,'m':.0042,'rho':1.205,'A':A}
    # D = 0.5*params['rho']*params['Cd']*params['A']
    # tf = 1000
    # nt = tf/params['dt']
    # i = 0
    # rk4 = RK4(derivs, params)
    # while vector[i,1] >=0 and i <= nt:
    #     vec_new = rk4(vector[i,:],t[i])
    #     # print(Xnew)
    #     vector = np.append(vector,vec_new,axis=0)
    #     t = np.append(t,np.array([t[i]+params['dt']]),axis=0)
    #     i += 1
    # plt.plot(vector[:,0],vector[:,1])
    # plt.show()
    # term_vel = np.sqrt(params['m']*params['g']/D)
    # print(f'Initial velocity: {vector[0,3]:.3f}, final velocity: {vector[-10,3]:.3f}')
    # print(f'Calculated terminal velocity: {term_vel:.3f}')

    def derivs(_, t, *args):
        return t**2

    dx = 0.01
    x = np.array([[-2]])
    xf = 2
    nx = int((xf-x[0,0])/dx)
    y = np.array([[(-2)**3/3]])
    params = {'dt':dx}
    rk4 = RK4(derivs, params)

    for i in range(nx):
        yn = rk4(y[i], x[i])
        xn = x[i] + dx
        y = np.append(y, yn,axis=0)
        x = np.append(x, np.array([xn]),axis=0)

    plt.plot(x, x**2, label='function')
    plt.plot(x, y, label='integral')
    plt.plot(x,x**3/3, label='exact',ls=':')
    plt.legend()
    plt.show()
