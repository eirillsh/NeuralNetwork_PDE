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
	def __init__(self, A, num_neurons=[], activation_functions=[], t=1e30):
		super().__init__()
		n = len(A)
		self.A = tf.convert_to_tensor(A)					# symmetric n x n matrix 
		self.I = tf.linalg.LinearOperatorIdentity(n, dtype='float64').to_dense()
		self.set_t(t)										# default t-value (should be inf)
		# Metoden under brude nok heller vært add_recurrent_layers....	
		#self.add_recurrent_layers(1, n, num_neurons, activation_functions)
		self.add_feed_forward_layers(1, n, num_neurons, activation_functions)

	# normalized eigenvector
	@property
	def x(self):
		xT = self(self.t)
		x = tf.transpose(xT)
		# normalization factor
		c = np.sqrt((xT@x)[0][0])
		return x/c
	
	# eigenvalue
	@property
	def E(self):
		xT = self(self.t)
		x = tf.transpose(xT)
		E = (xT@self.A@x)/(xT@x)
		return E[0][0]

	def MSE(self, exact_function):
		u_e = u_e(self.x, self.t)



	@tf.function
	def loss(self):
		'''
		Mean Squared Error, evaluating :
					 lim{t-> inf} x_t = -x + f(x)
		'''
		with tf.GradientTape(watch_accessed_variables=False) as Dt:
			Dt.watch(self.t)
			xT = self(self.t)
			x = tf.transpose(xT)
		Dt_x = Dt.jacobian(x, self.t)[:, 0, 0, 0]
		x_t = -x[:,0] + self.f(x)[:,0]
		return tf.losses.mean_squared_error(x_t, Dt_x)

	#f(x) = [x.T x A + (1 - x.T A x) I] x
	def f(self, x):
		xT = tf.transpose(x)
		return (xT@x*self.A + (1 - xT@self.A@x)*self.I)@x

	# asserting: Ax = Ex
	def error(self):
		x = self.x
		return self.A@x - self.E@x

	# set input for the network
	def set_t(self, t):
		self.t = self.array_to_tensor([float(t)])

	
