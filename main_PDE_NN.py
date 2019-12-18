import tensorflow as tf
import numpy as np
from heat_flow import HeatFlow
from heat_flow_solver import HeatFlowSolver, create_mesh
import matplotlib.pyplot as plt
from time import time
# -----SETTINGS FOR PLOTS-----
line_width = 3
font = 15
path = "Figures/PDE_NN_"


color_NN1 = 'orangered'
color_NN2 = 'cornflowerblue'
color_NN3 = 'forestgreen'
color_exact = 'darkgray'

show = False
t1 = 0.05
t2 = 0.75
t3 = 0.50
labels = [r"$NN_1$", r"$NN_2$", r"$NN_3$"]
# ---------------------------
eta = 0.01
epochs = 2000
num = 10 

u_e = HeatFlow(1, 1, 1)
u0 = lambda x: tf.sin(np.pi*x)

N = 10000
T = L = 1

NN1 = HeatFlowSolver(u0, N, T, num_neurons=[20]*2, activation_functions=["sigmoid"]*2)
NN2 = HeatFlowSolver(u0, N, T, num_neurons=[20]*2, activation_functions=["sigmoid"]*2)
NN3 = HeatFlowSolver(u0, N, T, num_neurons=[20]*2, activation_functions=["sigmoid"]*2)

print("first starting")
start_time = time()
loss_NN1 = NN1.solve_verbose(eta, epochs, num_epochs=num)
print(f"{float(time() - start_time)/60 :20.5f} min")
print("second starting")
start_time = time()
loss_NN2 = NN2.solve_verbose(eta, epochs, num_epochs=num)
print(f"{float(time() - start_time)/60 :20.5f} min")
print("third starting")
start_time = time()
loss_NN3 = NN3.solve_verbose(eta, epochs, num_epochs=num)
print(f"{float(time() - start_time)/60 :20.5f} min")
print()



Nx = 300
x = np.linspace(0, 1, Nx)
# Plotting t1
t = np.zeros(Nx) + t1
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
t = np.zeros(Nx) + t2
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

# Plotting t3
t = np.zeros(Nx) + t3
plt.plot(x, u_e(x, t3), color=color_exact, label="exact", linewidth=line_width)
plt.plot(x, NN1.u(x, t), "--", color=color_NN1, label=labels[0], linewidth=line_width)
plt.plot(x, NN2.u(x, t), "--", color=color_NN2, label=labels[1], linewidth=line_width)
plt.plot(x, NN3.u(x, t), "--", color=color_NN3, label=labels[2], linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$x$", fontsize=font)
plt.ylabel(r"$u(x, t=%.2f)$"%t3, fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "t3.pdf")
if show:
	plt.show()
else:
	plt.close()


# -------plotting error-------

Nt = 100
Nx = 100
time = np.linspace(0, 1, Nt)
x = np.linspace(0, 1, Nx)
# ----------------------------

# mean absolute error
max_abs_NN1 = np.zeros(Nt)
max_abs_NN2 = np.zeros(Nt)
max_abs_NN3 = np.zeros(Nt)
# mean relative error
max_rel_NN1 = np.zeros(Nt)
max_rel_NN2 = np.zeros(Nt)
max_rel_NN3 = np.zeros(Nt)
# mean squared error
MSE_NN1 = 0
MSE_NN2 = 0
MSE_NN3 = 0
for i in range(Nt):
	u = u_e(x, time[i])
	t = np.zeros(Nx) + time[i]
	# mean absolute error
	error_NN1 = np.abs(u - NN1.u(x, t))
	error_NN2 = np.abs(u - NN2.u(x, t))
	error_NN3 = np.abs(u - NN3.u(x, t))
	max_abs_NN1[i] = np.mean(error_NN1)
	max_abs_NN2[i] = np.mean(error_NN2)
	max_abs_NN3[i] = np.mean(error_NN3)
	# mean squared error
	MSE_NN1 += np.sum(error_NN1**2)
	MSE_NN2 += np.sum(error_NN2**2)
	MSE_NN3 += np.sum(error_NN3**2)
	# mean relative error
	error_NN1 = error_NN1[u != 0]
	error_NN2 = error_NN2[u != 0]
	error_NN3 = error_NN3[u != 0]
	u = u[u != 0]
	max_rel_NN1[i] = np.mean(error_NN1/u)
	max_rel_NN2[i] = np.mean(error_NN2/u)
	max_rel_NN3[i] = np.mean(error_NN3/u)

N = Nt*Nx
MSE_test_NN1 = MSE_NN1/N
MSE_test_NN2 = MSE_NN2/N
MSE_test_NN3 = MSE_NN3/N

# mean squared error on train data
MSE_train_NN1 = NN1.MSE(u_e)
MSE_train_NN2 = NN2.MSE(u_e)
MSE_train_NN3 = NN3.MSE(u_e)
print(f'{"":10}{"COST":19s}{"MSE(TRAIN)":22s}{"MSE(TEST)"}')
print(f"NN1: {loss_NN1:.7e}  {MSE_train_NN1:20.7e}  {MSE_test_NN1:20.7e}")
print(f"NN2: {loss_NN2:.7e}  {MSE_train_NN2:20.7e}  {MSE_test_NN2:20.7e}")
print(f"NN3: {loss_NN3:.7e}  {MSE_train_NN3:20.7e}  {MSE_test_NN3:20.7e}")


plt.plot(time, max_abs_NN1, color=color_NN1, label=labels[0], linewidth=line_width)
plt.plot(time, max_abs_NN2, color=color_NN2, label=labels[1], linewidth=line_width)
plt.plot(time, max_abs_NN3, color=color_NN3, label=labels[2], linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$t$", fontsize=font)
plt.ylabel(r"$\bar{E}(t)$", fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "error_absolute.pdf")
if show:
	plt.show()
else:
	plt.close()


plt.plot(time, max_rel_NN1, color=color_NN1, label=labels[0], linewidth=line_width)
plt.plot(time, max_rel_NN2, color=color_NN2, label=labels[1], linewidth=line_width)
plt.plot(time, max_rel_NN3, color=color_NN3, label=labels[2], linewidth=line_width)
plt.legend(fontsize=font)
plt.xlabel(r"$t$", fontsize=font)
plt.ylabel(r"$\bar{E}_{rel}(t)$", fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout()
plt.savefig(path + "error_relative.pdf")
if show:
	plt.show()
else:
	plt.close()
