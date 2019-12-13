import autograd.numpy as np
# Testing used autograd, and therefore requires numpy to be imported from autograd.


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
		self.ll = (self.w/c)**2

	def __call__(self, x, t):
		'''
		u(x, t) = F(x)G(t)
		'''
		return self.F(x)*self.G(t)

	def F(self, x):
		'''
		Separation of variables: x-dependent
		'''
		return np.sin(self.w*x)

	def G(self, t):
		'''
		Separation of variables: t-dependent
		'''
		return np.exp(-self.ll*t)

	def xx(self, x, t):
		'''
		u_xx : double derivative with respect to x
		'''
		return -self.w**2*self.F(x)*self.G(t)

	def t(self, x, t):
		'''
		u_t : derivative with respect to t
		'''
		return -self.ll*self.F(x)*self.G(t)

