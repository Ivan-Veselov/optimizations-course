from decimal import Decimal


def chebyshev(func, iterations_number, initial_point):
    point = (func.M() + func.m()) / (func.M() - func.m())

    x = initial_point
    prev_x = x

    alpha = Decimal(2) / (func.M() + func.m())
    beta = Decimal(0)
    polynom_values = [Decimal(1), point, Decimal(2) * point ** 2 - Decimal(1)]

    for _ in range(iterations_number):
        new_x = x - alpha * func.gradient(x) + beta * (x - prev_x)
        prev_x = x
        x = new_x

        alpha = polynom_values[1] / polynom_values[2] * Decimal(4) / (func.M() - func.m())
        beta = polynom_values[0] / polynom_values[2]
        polynom_values = polynom_values[1:] + \
                            [Decimal(2) * point * polynom_values[2] - polynom_values[1]]

    return x
