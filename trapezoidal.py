import numpy as np

class Trapezoidal:

    def __init__(self, return_array=True):
        self.return_array = return_array

    def _integrate(self, array, dx):
        ny = int(len(array))
        mults = np.ones(ny)*2
        mults[0], mults[1] = 1, 1
        if self.return_array:
            integral = dx/2 * np.cumsum(mults*array)
            return integral
        else:
            integral = dx/2 * np.sum(mults*array)
            return integral

    def __call__(self, array, dx):
        result = self._integrate(array, dx)
        return result

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x)
    dx = x[1]-x[0]
    trapz = Trapezoidal()
    exact = -np.cos(x)
    approx = trapz(y, dx)
    approx += exact[0]
    plt.plot(x, exact, label='exact')
    plt.plot(x, approx, label='numerical',ls='-.')
    plt.legend()
    plt.show()