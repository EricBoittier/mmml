import numpy as np

e0 = -2143.4256137550415
ep = -2143.425085940985
em = -2143.4256137550415
step = 0.0001
f_fd = -(ep - em) / (2.0 * step)
print(f"Computed FD in float32 garbage: {f_fd}")
