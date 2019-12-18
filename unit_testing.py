import autograd.numpy as np
from heat_flow import HeatFlow
from finite_difference import FiniteDifference
from main_eigen_problem import create_symmetric_matrix
from autograd import jacobian, hessian
import nose


def test_HeatFlow():
	'''
	Testing HeatFlow from heat_flow.py
	Exact solution of u_xx = c^2 u_t
	'''
	tol = 1e-12
	for n in [1, 2, 3]:
		for c in [0.33, 1, 1.5]:
			for L in [0.5, 1, 10]:
				u = HeatFlow(c, n, L)
				# setting up for testing that PDE is correct
				u_t = jacobian(u, 1)
				u_xx = hessian(u, 0)
				for t in [0.0, 0.05, 1.0]:
					# testing boundary conditions
					ut0, utL = u(0, t), u(L, t)
					msg = f"boundary conditions should be zero in HeatFlow, not ({ut0:.1g}, {utL:.1g})"
					assert abs(ut0) < tol and abs(utL) < tol, msg
					for x in np.linspace(0, L, 30):
						# testing that derivative is implemented correctly
						msg = "Derivative of HeatFlow with respect to t yielded unexpected answer"
						assert abs(u_t(x, t) - u.t(x, t)) < tol, msg
						msg = "Double derivative of HeatFlow with respect to x yielded unexpected answer"
						assert abs(u_xx(x, t) - u.xx(x, t)) < tol, msg
						# testing that PDE is correct
						ccut, uxx  = c*c*u.t(x, t), u.xx(x, t)
						msg = f"u in HeatFlow does not fulfill: u_xx = c^2 u_t -> {uxx:.7g} != {ccut:.7g}"
						assert abs(uxx - ccut)  < tol, msg


def test_FiniteDifference():
	'''
	Testing FiniteDifference from finite_difference.py
	Numerical solution to u_xx = c^2 u_t
	'''
	name = "FiniteDifference: "
	tol = 1e-12
	eps = 1e-16
	c = L = 1
	for n in [1, 3]:
		u_e = HeatFlow(c, n, L)
		u0 = u_e.F
		pde = FiniteDifference(c, u0, L)
		T = 1
		for dx in [0.05, 0.1, 0.2]:
			dt = dx*dx/4
			# Testing that they all yield same answers (hence the bad values for dx)
			x_full, t_full, u_full = pde.solve(dx, dt, T)
			x_final, t_final, u_final = pde.solve(dx, dt, T, sample="end")
			x_sample, t_sample, u_sample = pde.solve(dx, dt, T, sample=2)
			# testing that u(x, T) is same (not correct) for all three methods
			assert max(abs(u_full[-1] - u_final)) < eps, name+"solve methods full and final should yield same result"
			assert max(abs(u_full[-1] - u_sample[-1])) < eps, name+"solve methods full and sample should yield same result"
			# returned x-values should be the same
			assert sum(abs(x_full - x_final)) < eps, name+"solve methods full and final should return same x-values"
			assert sum(abs(x_full - x_sample)) < eps, name+"solve methods full and sample should return same x-values"
			# returned final T-value should be the same (dx, L and T chosen such that this should be true)
			assert abs(t_full[-1] - t_final) < eps, name+"solve methods full and final should return same final t-value"
			assert abs(t_full[-1] - t_sample[-1]) < eps, name+"solve methods full and sample should return same final t-value"
		# Comparing to the exact solution. Unrealistic values to compute over longer time, only for testing.
		# must have really small dx to get proper good results. dx = 10^{-3}
		# error accumulates, so T must also be small... (chosen T=10^{-4} yields 401 iterations)
		dx = 1e-3
		T = 1e-4
		dt = dx*dx/4
		x, t_full, u_full = pde.solve(dx, dt, T)
		x, t_final, u_final = pde.solve(dx, dt, T, sample="end")
		x, t_sample, u_sample = pde.solve(dx, dt, T, sample=2)
		# As any error would accumulate, we only test the last t-value
		error = max(abs(u_e(x, t_full[-1]) - u_full[-1])**2)
		msg = name+f"method for solving 'full' yields unexpected answer. max error found:{error:.7g}"
		assert error < tol, msg
		error = max(abs(u_e(x, t_sample[-1]) - u_sample[-1])**2)
		msg = name+f"method for solving 'sample' yields unexpected answer. max error found:{error:.7g}"
		assert error < tol, msg
		error = max(abs(u_e(x, t_final) - u_final)**2)
		msg = name+f"method for solving 'final' yields unexpected answer. max error found:{error:.7g}"
		assert error < tol, msg

def test_Eigenvalue():
	n = 6
	A = create_symmetric_matrix(n)
	msg = "create_symmetric_matrix function does not yield symmetric matrix"
	# Testing that input matrix to EigenProblem class is symmetric
	assert np.sum(np.abs(A - A.T)) < 1e-32, msg


nose.run()
