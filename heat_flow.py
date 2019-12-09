import numpy as np


class HeatFlow:
	''' 
	Solution to Partial Differential Equation:
			u_xx = c^2 u_t
	Boundary conditions:
		u(0, t) = u(L, t) = 0
	Only eigenfunctions, no linear combinations.
	'''
	def __init__(self, c, n, L):
		self.c = c
		self.L = L
		self.n = n
		self.w =  np.pi*n/L
		self.ll = (c*self.w)**2

	# u(x, t) = F(x)G(t)
	def __call__(self, x, t):
		return self.F(x)*self.G(t)

	# Separation of variables: x-dependent
	def F(self, x):
		return np.sin(self.w*x)

	# Separation of variables: t-dependent
	def G(self, t):
		return np.exp(-self.ll*t)

	# u_xx : double derivative with respect to x
	def xx(self, x, t):
		return -self.w**2*self.F(x)*self.G(t)

	# u_t : derivative with respect to t
	def t(self, x, t):
		return -self.ll*self.F(x)*self.G(t)

