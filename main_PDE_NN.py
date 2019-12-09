import tensorflow as tf
import numpy as np
from heat_flow_solver import HeatFlowSolver, create_mesh


u0 = lambda x: tf.sin(np.pi*x)

Nx = Nt = 10
T = 1
x_range, t_range = np.linspace(0, 1, 10), np.linspace(0, T, 10)
x, t = create_mesh(x_range, t_range)
u_e = np.sin(np.pi*x)*np.exp(-np.pi**2*t)

NN = HeatFlowSolver(u0, Nx, Nt, T, num_neurons=[50], activation_functions=["sigmoid"])


NN.solve_verbose(0.01, 10000, num_epochs=10, tol=1e-6)
u = NN.u(x, t)
diff = np.abs(u_e - u)
print("error:", np.mean(diff), np.max(diff))