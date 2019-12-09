import tensorflow as tf
import numpy as np
from heat_flow import HeatFlow
from heat_flow_solver import HeatFlowSolver, create_mesh
import matplotlib.pyplot as plt
# -----SETTINGS FOR PLOTS-----
line_width = 3
font = 15
path = "Figures/PDE_NN_"


color_NN1 = 'maroon'
color_NN2 = 'gold'
color_NN3 = 'darkgreen'
color_exact = 'darkgray'

show = True
t1 = 0.05
t2 = 0.5
labels = ["layers = 1", "layers = 2", "layers = 3"]
# ---------------------------
eta = 0.01
epochs = 10000

u_e = HeatFlow(1, 1, 1)
u0 = lambda x: tf.sin(np.pi*x)

Nx = Nt = 10
T = L = 1

x, t = create_mesh(np.linspace(0, L, Nx), np.linspace(0, T, Nt))

NN1 = HeatFlowSolver(u0, Nx, Nt, T, num_neurons=[10], activation_functions=["sigmoid"])
NN2 = HeatFlowSolver(u0, Nx, Nt, T, num_neurons=[10]*2, activation_functions=["sigmoid"]*2)
NN3 = HeatFlowSolver(u0, Nx, Nt, T, num_neurons=[10]*3, activation_functions=["sigmoid"]*3)

print("one")
NN1.solve_verbose(eta, epochs, num_epochs=10, tol=1e-4)
print("two")
NN2.solve_verbose(eta, epochs, num_epochs=10, tol=1e-4)
print("three")
NN3.solve_verbose(eta, epochs, num_epochs=10, tol=1e-4)

N = 300
x = np.linspace(0, 1, N)

# Plotting t1
t = np.zeros(N) + t1
plt.plot(x, u_e(x, t1), color=color_exact, label="exact", linewidth=line_width)
plt.plot(x, NN1.u(x, t), "--", color=color_NN1, label=labels[0], linewidth=line_width)
plt.plot(x, NN2.u(x, t), "--", color=color_NN2, label=labels[1], linewidth=line_width)
plt.plot(x, NN3.u(x, t), "--", color=color_NN3, label=labels[2], linewidth=line_width)
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
t = np.zeros(N) + t2
plt.plot(x, u_e(x, t2), color=color_exact, label="exact", linewidth=line_width)
plt.plot(x, NN1.u(x, t), "--", color=color_NN1, label=labels[0], linewidth=line_width)
plt.plot(x, NN2.u(x, t), "--", color=color_NN2, label=labels[1], linewidth=line_width)
plt.plot(x, NN3.u(x, t), "--", color=color_NN3, label=labels[2], linewidth=line_width)
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

'''
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
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
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
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "error_relative.pdf")
if show:
	plt.show()
else:
	plt.close()
'''