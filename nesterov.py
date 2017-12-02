from decimal import Decimal


def solve_quadratic(b, c):
    D = b ** 2 - Decimal(4) * c
    x1 = (-b - Decimal.sqrt(D)) / Decimal(2)
    x2 = (-b + Decimal.sqrt(D)) / Decimal(2)

    if Decimal(0.) < x1 < Decimal(1.):
        return x1

    if Decimal(0.) < x2 < Decimal(1.):
        return x2

    assert False


def nesterov(func, iterations_number, initial_point):
    x = initial_point
    alpha = Decimal(1) / Decimal(2)
    y = x

    for _ in range(iterations_number):
        new_x = y - func.gradient(y) / func.M()
        new_alpha = solve_quadratic(alpha - func.m() / func.M(), -alpha)

        beta = alpha * (Decimal(1) - alpha) / (alpha ** 2 + new_alpha)
        y = new_x + beta * (new_x - x)

        alpha = new_alpha
        x = new_x

    return x
