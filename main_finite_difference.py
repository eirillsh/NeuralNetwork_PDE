from finite_difference import FiniteDifference
from heat_flow import HeatFlow
import numpy as np
import matplotlib.pyplot as plt

line_width = 2
u_e = HeatFlow(1, 1, 1)
x_e = np.linspace(0, 1, 300)
u0 = u_e.F
pde = FiniteDifference(1, u0, 1)

T1 = 0.05
T2 = 0.5

x_01, t1, u_t1_01 = pde.solve(T1, 0.1, sample="end")
x_01, t2, u_t2_01 = pde.solve(T2, 0.1, sample="end")

x_001, t1, u_t1_001 = pde.solve(T1, 0.01, sample="end")
x_001, t2, u_t2_001 = pde.solve(T2, 0.01, sample="end")

u_t1 = u_e(x_e, t1)
u_t2 = u_e(x_e, t2)

exact_color = 'grey'
plt.plot(x_e, u_e(x_e, 0), color=exact_color, label=f"u(x, 0)")
plt.plot(x_e, u_t1, color=exact_color, label=f"u(x, {t1:0.2f})")
plt.plot(x_e, u_t2, color=exact_color, label=f"u(x, {t2:0.2f})")
plt.legend()
plt.show()

color_dx01 = 'teal'
color_dx001 = 'palevioletred'
color_exact = 'black'

plt.title("t = 0.05")
plt.plot(x_e, u_t1, color=color_exact, linewidth=line_width)
plt.plot(x_001, u_t1_001, "--", color=color_dx001, linewidth=line_width)
plt.plot(x_01, u_t1_01, "--", color=color_dx01, linewidth=line_width)
plt.show()

plt.title("t = 0.5")
plt.plot(x_e, u_t2, color=color_exact, linewidth=line_width)
plt.plot(x_001, u_t2_001, "--", color=color_dx001, linewidth=line_width)
plt.plot(x_01, u_t2_01, "--", color=color_dx01, linewidth=line_width)
plt.show()





