import numpy as np
import scipy

class Amatrix:
    '''
    The purpose of this class is to return updated A matrices for the MPC controller. Since the A matrix for the UAV
    is nonlinear, it contains two of the states from the state vector, and needs to be updated every time step. This
    class will allow a class instance to produce a new A matrix given the current state vector.
    '''

    def __init__(self, params):
        self.params = params

    def _create_matrix(self, state):
        Ixx = self.params['Ixx']
        Iyy = self.params['Iyy']
        Izz = self.params['Izz']
        J = self.params['J']
        omega = self.params['omega']
        phi_dot = state[1]
        theta_dot = state[3]

        A = np.array([[0,1,0,0,0,0],
                      [0,0,0,J*omega/Ixx, 0, (Iyy-Izz)/Ixx*theta_dot],
                      [0,0,0,1,0,0],
                      [0,-J*omega/Iyy,0,0,0,(Izz-Ixx)/Iyy*phi_dot],
                      [0,0,0,0,0,1],
                      [0,(Ixx-Izz)/Izz*theta_dot/2,0,(Ixx-Iyy)/Izz*theta_dot/2,0,0]])
        
        return A

    def __call__(self, state):
        A = self._create_matrix(state)
        return A
