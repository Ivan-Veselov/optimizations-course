from decimal import Decimal

import numpy as np

import commons
from chebyshev import chebyshev
from heavy_ball import heavy_ball
from nesterov import nesterov


class Function(commons.Function):
    def __call__(self, arg):
        return Decimal(1) / Decimal(2) * (arg[0] ** 2 + Decimal(69) * arg[1] ** 2)

    def gradient(self, arg):
        return np.array([arg[0], Decimal(69) * arg[1]])

    def m(self):
        return Decimal(1)

    def M(self):
        return Decimal(69)


def main():
    iterations_number = int(input('Enter number of iterations: '))
    initial_point = np.array([Decimal(10.), Decimal(10.)])

    print('Heavy ball:', heavy_ball(Function(), iterations_number, initial_point))
    print('Nesterov:', nesterov(Function(), iterations_number, initial_point))
    print('Chebyshev:', chebyshev(Function(), iterations_number, initial_point))


if __name__ == "__main__":
    main()
