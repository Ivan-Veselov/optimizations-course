from decimal import Decimal


def heavy_ball(func, iterations_number, initial_point):
    alpha = Decimal(4) / (Decimal.sqrt(func.M()) + Decimal.sqrt(func.m())) ** 2
    beta = (Decimal.sqrt(func.M()) - Decimal.sqrt(func.m())) /\
                (Decimal.sqrt(func.M()) + Decimal.sqrt(func.m()))

    x = initial_point
    prev_x = x

    for _ in range(iterations_number):
        new_x = x - alpha * func.gradient(x) + beta * (x - prev_x)
        prev_x = x
        x = new_x

    return x
