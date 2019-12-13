from finite_difference import FiniteDifference
from heat_flow import HeatFlow
import numpy as np
import matplotlib.pyplot as plt

# -----SETTINGS FOR PLOTS-----
line_width = 3
font = 15
path = "Figures/finite_"


color_dx01 = 'crimson'
color_dx001 = 'teal'
color_exact = 'darkgray'

show = True
T1 = 0.05
T2 = 0.75
# ---------------------------

get_dt =  lambda dx: dx*dx/2

u_e = HeatFlow(1, 1, 1)
x_e = np.linspace(0, 1, 300)
u0 = u_e.F
pde = FiniteDifference(1, u0, 1)

dx = 0.1 
dt = get_dt(dx)
x_01, t1, u_t1_01 = pde.solve(dx, dt, T1, sample="end")
x_01, t2, u_t2_01 = pde.solve(dx, dt, T2, sample="end")

dx = 0.01 
dt = get_dt(dx)
x_001, t1, u_t1_001 = pde.solve(dx, dt, T1, sample="end")
x_001, t2, u_t2_001 = pde.solve(dx, dt, T2, sample="end")

u_t1 = u_e(x_e, t1)
u_t2 = u_e(x_e, t2)


# Plotting exact: the two chosen times + t=0
plt.plot(x_e, u_e(x_e, 0), color="black", label=r"$t = 0.00$", linewidth=line_width)
plt.plot(x_e, u_t1, color="darkmagenta", label=r"$t = %.2f$"%t1, linewidth=line_width)
plt.plot(x_e, u_t2, color="MediumPurple", label=r"$t = %.2f$"%t2, linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$x$", fontsize=font)
plt.ylabel(r"$u(x, t)$", fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
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
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "t1.pdf")
if show:
	plt.show()
else:
	plt.close()

# Plotting t2
plt.plot(x_e, u_t2, color=color_exact, label="exact", linewidth=line_width)
plt.plot(x_001, u_t2_001, "--", color=color_dx001, label=r"$\Delta x = 0.01$", linewidth=line_width)
plt.plot(x_01, u_t2_01, "--", color=color_dx01, label=r"$\Delta x = 0.1$", linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$x$", fontsize=font)
plt.ylabel(r"$u(x, t=%.2f)$"%t2, fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "t2.pdf")
if show:
	plt.show()
else:
	plt.close()


# -------plotting error-------

T = 1
dx = 0.1 
dt = get_dt(dx)
print(f"dt: {dt} = ", end="")
x_01, t_01, u_01 = pde.solve(dx, dt, T, sample=201)
print(f" {pde.dt} ?")
dx = 0.01 
dt = get_dt(dx)
print(f"dt: {dt} = ", end="")
x_001, t_001, u_001 = pde.solve(dx, dt, T, sample=401)
print(f" {pde.dt} ?\n\n")


# mean absolute error
abs_01 = np.zeros(len(t_01))
abs_001 = np.zeros(len(t_001))
# mean relative error
rel_01 = np.zeros(len(t_01))
rel_001 = np.zeros(len(t_001))
# mean squared error
MSE_01, MSE_001  = 0, 0

for i in range(len(t_01)):
	u = u_e(x_01,  t_01[i])
	error = abs(u - u_01[i])
	# mean squared error
	MSE_01 += np.sum(error**2)
	# mean abolute error
	abs_01[i]  = np.mean(error)
	# mean relative error
	expected = u[u != 0]
	computed = u_01[i, u!= 0]
	rel_01[i]  = np.mean(abs(expected - computed)/expected)
MSE_01 = MSE_01/(len(x_01)*len(t_01))

for i in range(len(t_001)):
	u = u_e(x_001,  t_001[i])
	error = abs(u - u_001[i])
	# mean squared error
	MSE_001 += np.sum(error**2)
	# mean abolute error
	abs_001[i]  = np.mean(error)
	# mean relative error
	expected = u[u != 0]
	computed = u_001[i, u!= 0]
	rel_001[i]  = np.mean(abs(expected - computed)/expected)
MSE_001 = MSE_001/(len(x_001)*len(t_001))


print(f"dx = 0.1  : MSE {MSE_01:.7g}")
print(f"dx = 0.01 : MSE {MSE_001:.7g}")


plt.plot(t_001, abs_001,  color=color_dx001, label=r"$\Delta x = 0.01$", linewidth=line_width)
plt.plot(t_01, abs_01, color=color_dx01, label=r"$\Delta x = 0.1$", linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$t$", fontsize=font)
plt.ylabel("mean absolute error", fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "error_absolute.pdf")
if show:
	plt.show()
else:
	plt.close()

plt.plot(t_001, rel_001,  color=color_dx001, label=r"$\Delta x = 0.01$", linewidth=line_width)
plt.plot(t_01, rel_01, color=color_dx01, label=r"$\Delta x = 0.1$", linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$t$", fontsize=font)
plt.ylabel("mean relative error", fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "error_relative.pdf")
if show:
	plt.show()
else:
	plt.close()



