import numpy as np
from math import ceil

class FiniteDifference:
	'''
	Class for Solving Partial Differential Equation:
					u_xx = c^2 u_t
	Often called the Heat Flow Equation. Solve for:
				 	u(x, t),      
							-> x in [0, L]  
							-> t in [0, inf)
	using Finite Difference.
	Centered difference in space.
	Forward difference in time.
	'''
	def __init__(self, c, initial_condition, L):
		self.u0 = initial_condition
		self.ut_pos0 = self.u0(0)
		self.ut_posL = self.u0(L)
		self.L = L
		self.c = c


	def solve(self, dx, dt, T, sample="full"):
		'''
		Solve heat flow equation using finite difference
		
		sample:
			"full"           : return all calulated points (default)
			int sample_size  : return sample_size of uniform t-points
			"final"			 : only return x-array for last iteration, final t
		'''
		if isinstance(sample, str):
			sample = sample.lower()
			if sample == "full":
				return self.solve_full(dx, dt, T)
			elif sample in ["end", "final"]:
				return self.solve_final(dx, dt, T)
			else:
				raise Error("Invalid choice for sample")
		elif isinstance(sample, int):
			if sample == 1:
				return self.solve_full(dx, dt, T)
			elif sample > 1:
				return self.solve_sample(sample, dx, dt, T)
			else:
				raise Error("Invalid choice for sample")
		else:
			raise Error("Invalid choice for sample")


	def solve_full(self, dx, dt, T):
		'''
		solve PDE and return u for all time steps
		'''
		x, M = self.set_position_interval(dx)
		N = self.set_time_interval(dt, T)
		t = np.linspace(0, T, N+1)

		#inital condition
		u = np.zeros((N+1, M+1))
		u[0] = self.u0(x)

		# boundary condition
		u[:, 0], u[:,-1] = self.ut_pos0, self.ut_posL

		# iterative scheme
		const = self.get_constant()
		for i in range(N):
			u[i+1, 1:M] = u[i, 1:M] + const*(u[i, 2:M+1] - 2*u[i, 1:M] + u[i, :M-1])

		return x, t, u


	def solve_final(self, dx, dt, T):
		'''
		solve PDE and return u for final time step
		'''
		x, M = self.set_position_interval(dx)
		N = self.set_time_interval(dt, T)

		#inital condition & boundary condition
		u = self.u0(x)

		# iterative scheme
		const = const = self.get_constant()
		for i in range(N):
			u[1:M] = u[1:M] + const*(u[2:M+1] - 2*u[1:M] + u[:M-1])
		return x, T, u


	def solve_sample(self, samples, dx, dt, T):
		'''
		solve PDE and return u for some time steps
		'''
		x, M = self.set_position_interval(dx)
		N = self.set_time_interval(dt, T, samples=samples)
		t = np.linspace(0, T, samples)
		skip = int(N/(samples-1)) - 1	# number of time calculations not to save
		
		#inital condition
		u = np.zeros((samples, M+1))
		u[0] = self.u0(x)

		# boundary condition
		u[:, 0], u[:,-1] = self.ut_pos0, self.ut_posL

		# iterative scheme
		const = self.get_constant()
		for i in range(samples - 1):
			u[i+1, 1:M] = u[i, 1:M] + const*(u[i, 2:M+1] - 2*u[i, 1:M] + u[i, :M-1])
			for j in range(skip):
				u[i+1, 1:M] = u[i+1, 1:M] + const*(u[i+1, 2:M+1] - 2*u[i+1, 1:M] + u[i+1, :M-1])
		return x, t, u


	def set_position_interval(self, dx):
		'''
		set interval for x. L 
		'''
		M = ceil(self.L/dx)
		x = np.linspace(0, self.L, M+1)
		self.dx = x[1]
		return x, M


	def set_time_interval(self, dt, T, samples=None):
		'''
		set interval for t. 
		'''
		N = ceil(T/dt)
		if samples != None:
			N = ceil(N/(samples-1))*(samples-1)
		self.dt = T/N
		return N


	def get_constant(self):
		'''
		constant used in iterative scheme
		'''
		return self.dt/(self.c*self.dx)**2



