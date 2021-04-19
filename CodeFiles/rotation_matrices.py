import numpy as np
#from sympy_transfer_mat import TransferMatrix

class RotationMatrix:

    ''' 
    The aim of this class is to produce a 3D rotation matrix in cartesian coordinates, based on a user-specified order of 
    inertial frame axes about which to rotate. 

    Parameters
    ----------
    mat_order: list of strings. A list of the order of axes to rotate about (i.e ['x','y','z']). The first element will be the first
    axis rotated about, and so on. 

    invert: boolean. The default value of false means the output of the rotation matrix will be to go from the body frame to the
    inertial frame. Setting invert=True means translating from the inertial frame to the body frame. These are inverse operations
    of each other.
    '''

    def __init__(self, mat_order, invert=False):
        self.mat_order = mat_order
        self.invert = invert

    def _Rx(self, a):
        row1 = np.array([1,0,0])
        row2 = np.array([0, np.cos(a), -np.sin(a)])
        row3 = np.array([0, np.sin(a), np.cos(a)])
        if self.invert:
            return np.array([row1, row2, row3]).T
        else:
            return np.array([row1, row2, row3])

    def _Ry(self, a):
        row1 = np.array([np.cos(a), 0, np.sin(a)])
        row2 = np.array([0, 1, 0])
        row3 = np.array([-np.sin(a), 0, np.cos(a)])
        if self.invert:
            return np.array([row1, row2, row3]).T
        else:
            return np.array([row1, row2, row3])

    def _Rz(self, a):
        row1 = np.array([np.cos(a), -np.sin(a), 0])
        row2 = np.array([np.sin(a), np.cos(a), 0])
        row3 = np.array([0, 0, 1])
        if self.invert:
            return np.array([row1, row2, row3]).T
        else:
            return np.array([row1, row2, row3])

    def __call__(self, angles, vector):
        ''' Call method for executing the rotation matrix.
        Inputs
        ------
        angles: list of floats. The angles in radians through which rotation will occur. The angles are assumed to be in the same 
        order as the rotation order specified in mat_order when instantiating an object of the RotationMatrix class.

        vector: array. The vector to be translated from the body fram to the inertial frame (ordered x-y-z).

        Returns an array that is the result of applying the rotation matrix to the specified input vector and the Z-Y-X rotation 
        transfer matrix. The returned vector's components are x-y-z.
        Note: Specifying a 3x3 identify matrix as the input vector will result in the first returned argument to be the 
        rotation matrix itself.
        '''
        func_dict = {'x':self._Rx, 'y':self._Ry, 'z':self._Rz}
        if self.invert:
            order = self.mat_order
        else:
            order = self.mat_order[::-1]
            angles = angles[::-1]
        matrix_order = []
        for axis in order:
            for k, v in func_dict.items():
                if k == axis:
                    matrix_order.append(v)
        matrix_prod = np.eye(3)
        for idx, mat in enumerate(matrix_order):
            matrix_prod = matrix_prod@mat(angles[idx])

        # T = TransferMatrix()
        # T_lam = T(angles)
        
        return matrix_prod@vector#, T_lam

    def __repr__(self):
        rep = f'RotationMatrix(mat_order: {self.mat_order}, invert: {self.invert}'
        return rep

# if __name__ == '__main__':
#     order = ['z','y','x']
#     angles = [np.pi/2,np.pi/2,0]
#     vec = np.array([0,0,-9.81])
#     R = RotationMatrix(order)
#     R_inv = RotationMatrix(order,invert=True)
#     new_vec = R(angles, vec)
#     inv_vec = R_inv(angles,new_vec)
#     print(new_vec)
#     print(inv_vec)




