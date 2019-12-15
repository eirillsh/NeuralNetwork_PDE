import tensorflow as tf
from neural_network import NeuralNetwork
import numpy as np

class EigenProblem(NeuralNetwork):
	'''
	Inherits form NerualNetwork, built on keras (tensorflow) layers.
	Neural Network for solving eigen problem:
				Ax = Ex,
	for real symmetric matrix A.
	'''
	def __init__(self, A, num_neurons=[], activation_functions=[], t=100):
		super().__init__(1, n, num_neurons, activation_functions, "linear")
		n = len(A)
		self.A = tf.convert_to_tensor(A)					# symmetric n x n matrix
		self.I = tf.linalg.LinearOperatorIdentity(n, dtype='float64').to_dense()
		self.set_t(t)										# default t-value (should be inf)


	@property
	def x(self):
		'''
		normalized eigenvector
		'''
		xT = self(self.t)
		x = tf.transpose(xT)
		c = np.sqrt((xT@x)[0][0])
		return x/c


	@property
	def E(self):
		'''
		eigenvalue
		'''
		xT = self(self.t)
		x = tf.transpose(xT)
		E = (xT@self.A@x)/(xT@x)
		return E[0][0]


	def loss(self):
		'''
		Mean Squared Error, evaluating :
					x_t = -x + f(x)
		'''

		with tf.GradientTape(watch_accessed_variables=False) as Dt:
			Dt.watch(self.t)
			xT = self(self.t)
			x = tf.transpose(xT)

		Dt_x = Dt.jacobian(x, self.t)[:, 0, 0, 0]
		x_t = -x[:,0] + self.f(x)[:,0]
		return tf.losses.mean_squared_error(x_t, Dt_x)


	def f(self, x):
		xT = tf.transpose(x)
		return (xT@x*self.A + (1 - xT@self.A@x)*self.I)@x


	def error(self):
		'''
		asserting: Ax = Ex
		'''
		x = self.x
		return self.A@x - self.E@x


	def set_t(self, t):
		'''
		set input for the network
		'''
		self.t = self.array_to_tensor([float(t)])
