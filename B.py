from decimal import Decimal

import numpy as np

import commons
from nesterov import nesterov


class Function(commons.Function):
    def __init__(self):
        self._M = Decimal(0)
        self._m = Decimal(0)

        pairs = np.vectorize(lambda x: Decimal(x))(np.random.uniform(-2, 2, (10000, 2, 2)))

        for pair in pairs:
            x, y = pair
            self._M = self._M.max(np.linalg.norm(self.gradient(x) - self.gradient(y)) /
                                  np.linalg.norm(x - y))

        self._m = self._M
        for pair in pairs:
            x, y = pair
            self._m = self._m.min(np.dot(self.gradient(x) - self.gradient(y), x - y) /
                                  np.linalg.norm(x - y) ** 2)

    def __call__(self, arg):
        return Decimal.exp(arg[0] + Decimal(3) * arg[1] - Decimal(0.1)) + \
               Decimal.exp(arg[0] - Decimal(3) * arg[1] - Decimal(0.1)) + \
               Decimal.exp(-arg[0] - Decimal(0.1))

    def gradient(self, arg):
        return np.array(
            [Decimal.exp(arg[0] + Decimal(3) * arg[1] - Decimal(0.1)) +
             Decimal.exp(arg[0] - Decimal(3) * arg[1] - Decimal(0.1)) -
             Decimal.exp(-arg[0] - Decimal(0.1)),
             3 * (Decimal.exp(arg[0] + Decimal(3) * arg[1] - Decimal(0.1)) -
                  Decimal.exp(arg[0] - Decimal(3) * arg[1] - Decimal(0.1)))])

    def m(self):
        return self._m

    def M(self):
        return self._M


def main():
    iterations_number = int(input('Enter number of iterations: '))
    initial_point = np.array([Decimal(2.), Decimal(1.)])

    print(nesterov(Function(), iterations_number, initial_point))
    print('Actual answer:', np.array([-Decimal.ln(Decimal(2)) / Decimal(2), Decimal(0)]))


if __name__ == "__main__":
    main()
