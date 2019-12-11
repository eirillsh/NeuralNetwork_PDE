import tensorflow as tf
import numpy as np 
from neural_network import NeuralNetwork


class HeatFlowSolver(NeuralNetwork):
	''' 
	Inherits form NerualNetwork, built on keras (tensorflow) layers.
	Neural Network for Solving Partial Differential Equation:
					u_xx = c^2 u_t
	Often called the Heat Flow Equation. Solve for:
				 	u(x, t)      
							-> x in [0, L]  
							-> t in [0, inf)
	'''
	def __init__(self, u0, N, T, num_neurons=[], activation_functions=[], L=1, c=1):
		super().__init__()
		self.u0 = u0								# initial condition
		self.cc = tf.constant(c*c, dtype='float64')	# constant in PDE
		self.L = tf.constant(L, dtype='float64')	# upper bound on x
		# Erstatte med random tall??
		x, t = create_random_x_t(N, L, T)
		self.x, self.t = self.array_to_tensor(x), self.array_to_tensor(t)
		self.add_feed_forward_layers(2, 1, num_neurons, activation_functions, "linear")

	# solution function, used for predicting
	def u(self, x, t):
		x, t = self.array_to_tensor(x), self.array_to_tensor(t)
		return self.trial_function(x, t).numpy()[:, 0]

	# trial function, used when training
	def trial_function(self, x, t):
		X = tf.concat([x, t], 1)
		return self.u0(x) + x*(self.L - x)*t*self(X, training=False)

	# mean squared error 
	def MSE(self, exact_function):
		u_e = exact_function(self.x, self.t)
		u = self.trial_function(self.x, self.t).numpy()[:, 0]
		return np.mean((u_e - u)**2)

	# MSE: u_xx = c^2 u_t
	#@tf.function
	def loss(self):
		with tf.GradientTape(watch_accessed_variables=False) as velocity:
			velocity.watch(self.x)
			with tf.GradientTape(watch_accessed_variables=False) as position:
				position.watch(self.x)
				with tf.GradientTape(watch_accessed_variables=False) as time:
					time.watch(self.t)
					u = self.trial_function(self.x, self.t)
				u_t = time.gradient(u, self.t)
			u_x = position.gradient(u, self.x)
		u_xx = velocity.gradient(u_x, self.x)
		return tf.losses.mean_squared_error(u_xx[:, 0], self.cc*u_t[:, 0])



def create_mesh(x_range, t_range):
	'''
	creates and returns 1D-arrays of x and t
	with the combinations of given x-array and t-array
	'''
	N, M = len(t_range), len(x_range)
	x, t = np.zeros(N*M), np.zeros(N*M)
	for i in range(N):
		start = i*M
		stop = start + M
		x[start:stop] = x_range
		t[start:stop] = t_range[i]
	return x, t


def create_random_x_t(N, L, t_final):
	'''
	creates and returns 1D-arrays of x and t
	with random numbers t in [0, t_final] and x in [0, L]
	'''
	x, t = L*np.random.ranf(N), t_final*np.random.ranf(N)
	t[0:5] = 0    		# make sure to include initial time
	t[5:15] = t_final  	# make sure to include final time
	x[0:10] = 0 		# make sure to include edge cases several times
	x[10:20] = L 		# make sure to include edge cases several times
	np.random.shuffle(x), np.random.shuffle(t)
	return x, t


