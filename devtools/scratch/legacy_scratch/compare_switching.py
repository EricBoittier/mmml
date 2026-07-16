import numpy as np

def cubic_switch(s):
    return s * s * (3.0 - 2.0 * s)

def quintic_switch(s):
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))

def calc_derivatives(func, s):
    ds = 1e-5
    f = func(s)
    f_plus = func(s + ds)
    f_minus = func(s - ds)
    first_deriv = (f_plus - f_minus) / (2 * ds)
    second_deriv = (f_plus - 2 * f + f_minus) / (ds * ds)
    return f, first_deriv, second_deriv

print("s      | Cubic F'' | Quintic F''")
print("---------------------------------")
for s in [0.0, 0.0001, 0.5, 0.9999, 1.0]:
    _, _, c2 = calc_derivatives(cubic_switch, s)
    _, _, q2 = calc_derivatives(quintic_switch, s)
    print(f"{s:<6} | {c2:>9.2f} | {q2:>11.2f}")

