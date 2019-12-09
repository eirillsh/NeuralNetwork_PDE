from finite_difference import FiniteDifference
from heat_flow import HeatFlow
import numpy as np
import matplotlib.pyplot as plt

# -----SETTINGS FOR PLOTS-----
line_width = 2
font = 15
path = "Figures/finite_"


color_dx01 = 'crimson'
color_dx001 = 'teal'
color_exact = 'grey'

show = False
# ---------------------------

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

# Plotting exact: the two chosen times + t=0
plt.plot(x_e, u_e(x_e, 0), color="black", label=r"$t = 0.00$", linewidth=line_width)
plt.plot(x_e, u_t1, color="darkmagenta", label=r"$t = %.2f$"%t1, linewidth=line_width)
plt.plot(x_e, u_t2, color="slateblue", label=r"$t = %.2f$"%t2, linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$x$", fontsize=font)
plt.ylabel(r"$u(x, t)$", fontsize=font)
plt.tight_layout()
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.savefig(path + "exact.pdf")
if show:
	plt.show()
else:
	plt.close()

# Plotting t1
plt.plot(x_e, u_t1, color=color_exact, label="exact", linewidth=line_width)
plt.plot(x_001, u_t1_001, "--", color=color_dx001, label=r"$\Delta x = 0.01$", linewidth=line_width)
plt.plot(x_01, u_t1_01, "--", color=color_dx01, label=r"$\Delta x = 0.1$", linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$x$", fontsize=font)
plt.ylabel(r"$u(x, t=%.2f)$"%t1, fontsize=font)
plt.tight_layout()
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.savefig(path + "t1.pdf")
if show:
	plt.show()
else:
	plt.close()

# Plotting t2
plt.plot(x_e, u_t2, color=color_exact, linewidth=line_width)
plt.plot(x_001, u_t2_001, "--", color=color_dx001, label=r"$\Delta x = 0.01$", linewidth=line_width)
plt.plot(x_01, u_t2_01, "--", color=color_dx01, label=r"$\Delta x = 0.1$", linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$x$", fontsize=font)
plt.ylabel(r"$u(x, t=%.2f)$"%t2, fontsize=font)
plt.tight_layout()
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.savefig(path + "t2.pdf")
if show:
	plt.show()
else:
	plt.close()


# -------plotting error-------

T = 1
x_01, t_01, u_01 = pde.solve(T, 0.1)
x_001, t_001, u_001 = pde.solve(T, 0.01, sample=400)


# max absolute error
max_01 = np.zeros(len(t_01))
max_001 = np.zeros(len(t_001))
for i in range(len(t_01)):
	max_01[i]  = np.mean(np.abs(u_e(x_01,  t_01[i]) - u_01[i]))
for i in range(len(t_001)):
	max_001[i] = np.mean(np.abs(u_e(x_001, t_001[i]) - u_001[i]))

plt.plot(t_001, max_001,  color=color_dx001, label=r"$\Delta x = 0.01$", linewidth=line_width)
plt.plot(t_01, max_01, color=color_dx01, label=r"$\Delta x = 0.1$", linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$t$", fontsize=font)
plt.ylabel("mean absolute error", fontsize=font)
plt.tight_layout()
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.savefig(path + "error_absolute.pdf")
if show:
	plt.show()
else:
	plt.close()

# mean relative error
max_01 = np.zeros(len(t_01))
for i in range(len(t_01)):
	u = u_e(x_01,  t_01[i])
	expected = u[u != 0]
	computed = u_01[i, u!= 0]
	max_01[i]  = np.mean(abs(expected - computed)/expected)

max_001 = np.zeros(len(t_001))
for i in range(len(t_001)):
	u = u_e(x_001, t_001[i])
	expected = u[u != 0]
	computed = u_001[i, u!= 0]
	max_001[i] = np.mean(abs(expected - computed)/expected)

plt.plot(t_001, max_001,  color=color_dx001, label=r"$\Delta x = 0.01$", linewidth=line_width)
plt.plot(t_01, max_01, color=color_dx01, label=r"$\Delta x = 0.1$", linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$t$", fontsize=font)
plt.ylabel("mean relative error", fontsize=font)
plt.tight_layout()
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.savefig(path + "error_relative.pdf")
if show:
	plt.show()
else:
	plt.close()



