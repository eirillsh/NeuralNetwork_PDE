import numpy as np

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

	# solve PDE
	def solve(self, T, dx, dt="dt", sample="full"):
		if isinstance(sample, str):
			sample = sample.lower()
			if sample == "full":
				return self.solve_full(T, dx, dt)
			elif sample in ["end", "final"]:
				return self.solve_final(T, dx, dt)
			else:
				raise Error("Invalid choice for sample")
		elif isinstance(sample, int):
			if sample == 1:
				return self.solve_full(T, dx, dt)
			elif sample > 1:
				return self.solve_sample(sample, T, dx, dt)
			else:
				raise Error("Invalid choice for sample")
		else:
			raise Error("Invalid choice for sample")

	# solve PDE and return u for all time steps
	def solve_full(self, T, dx, dt="dt"):
		x, M = self.get_x(dx)
		N = self.get_N(T, dt, self.dx)
		t = np.linspace(0, T, N+1)
		self.dt = t[1]

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

	# solve PDE and return u for final time step
	def solve_final(self, T, dx, dt="dt"):
		x, M = self.get_x(dx)
		N = self.get_N(T, dt, self.dx)
		self.dt = T/N

		#inital condition
		u = self.u0(x)

		# boundary condition
		u[0], u[-1] = self.ut_pos0, self.ut_posL

		# iterative scheme
		const = const = self.get_constant()
		for i in range(N):
			u[1:M] += const*(u[2:M+1] - 2*u[1:M] + u[:M-1])
		return x, T, u

	# solve PDE and return u for some time steps
	def solve_sample(self, samples, T, dx, dt="dt"):
		x, M = self.get_x(dx)
		N = self.get_N(T, dt, self.dx)
		N -= int(N) % int(samples-1)
		self.dt = T/N
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
				u[i+1, 1:M] += const*(u[i+1, 2:M+1] - 2*u[i+1, 1:M] + u[i+1, :M-1])
		return x, t, u

	# get x interval and length 
	def get_x(self, dx):
		M = int(self.L/dx)
		x = np.linspace(0, self.L, M+1)
		self.dx = x[1]
		return x, M

	# compute number of time steps
	def get_N(self, T, dt, dx):
		if dt == "dt":
			dt = dx*dx/4
		N = int(T/dt) 
		return N

	# get constant used in PDE
	def get_constant(self):
		return self.dt*(self.c/self.dx)**2



