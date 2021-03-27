import sympy
import numpy as np
sympy.init_printing()

class TransferMatrix:
    ''' This transfer matrix is designed specifically for the Z-Y-X rotation order '''
    def _Rx(self,a):
        row1 = sympy.Matrix([1, 0, 0])
        row2 = sympy.Matrix([0, sympy.cos(a), -sympy.sin(a)])
        row3 = sympy.Matrix([0, sympy.sin(a), sympy.cos(a)])
        return sympy.Matrix([row1, row2, row3]).reshape(3,3)

    def _Ry(self,a):
        row1 = sympy.Matrix([sympy.cos(a), 0, sympy.sin(a)])
        row2 = sympy.Matrix([0, 1, 0])
        row3 = sympy.Matrix([-sympy.sin(a), 0, sympy.cos(a)])
        return sympy.Matrix([row1, row2, row3]).reshape(3,3)

    def _Rz(self,a):
        row1 = sympy.Matrix([sympy.cos(a), -sympy.sin(a), 0])
        row2 = sympy.Matrix([sympy.sin(a), sympy.cos(a), 0])
        row3 = sympy.Matrix([0, 0, 1])
        return sympy.Matrix([row1, row2, row3]).reshape(3,3)

    def _pos_matrix(self,pos):
        ''' place a one on the specified diagonal '''
        mat = sympy.zeros(3)
        mat[pos,pos] = 1
        return mat

    def __call__(self, angles):
        ''' Angles must be in Z-Y-X order (psi, theta, phi) '''
        x_mat = self._pos_matrix(0)
        y_mat = self._pos_matrix(1)
        z_mat = self._pos_matrix(2)
        phi, theta, psi = sympy.symbols('phi theta psi')
        rx = self._Rx(phi)
        ry = self._Ry(theta)
        rz = self._Rz(psi)
        ryx = ry@rx
        T_inv = x_mat + rx.T@y_mat + ryx.T@z_mat
        T = T_inv.inv()
        T_lam = sympy.lambdify([psi,theta,phi],T, 'numpy')
        return T_lam(*angles)


if __name__ == '__main__':
    # x_mat = pos_matrix(0)
    # y_mat = pos_matrix(1)
    # z_mat = pos_matrix(2)
    # phi, theta, psi = sympy.symbols('phi theta psi')
    # rx = Rx(phi)
    # ry = Ry(theta)
    # rz = Rz(psi)
    # ryx = ry@rx
    # T_inv = x_mat + rx.T@y_mat + ryx.T@z_mat
    # T = T_inv.inv()
    # print(sympy.simplify(T))
    angles = [0,0,0]
    xfer = TransferMatrix()
    print(xfer(angles))