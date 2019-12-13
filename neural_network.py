import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, SimpleRNN
from tensorflow.keras.optimizers import SGD as GradientDecent
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'		# prevents some complaining


class NeuralNetwork(tf.keras.Sequential):
	'''
	Abstract artificial neural network built on keras (tensorflow) layers.
	Back propagation using gradient decent.
	'''
	def __init__(self):
		super().__init__()


	def add_feed_forward_layers(self, features, size_output, num_neurons, activation_functions, output_activaton):
		'''
		create layers for feed forward neural network
		'''
		dtype = 'float64'					#default is float32
		init = initializers.glorot_normal()
		# adding input layer
		self.add(InputLayer(input_shape=(features,), dtype=dtype))
		# adding hidden layers
		for neurons, f in zip(num_neurons, activation_functions):
			self.add(Dense(neurons, activation=f, dtype=dtype, kernel_initializer=init))
		# adding output layer
		self.add(Dense(size_output, dtype=dtype, activation=output_activaton))


	def add_recurrent_layers(self, features, size_output, num_neurons, activation_functions, output_activaton):
		'''
		create layers for recurrent neural network
		'''
		dtype = 'float64'					#default is float32
		# adding input layer
		self.add(InputLayer(input_shape=(features,), dtype=dtype))
		# adding hidden layers
		for neurons, f in zip(num_neurons, activation_functions):
			self.add(SimpleRNN(neurons, activation=f, dtype=dtype))
		# adding output layer
		self.add(SimmpleRNN(size_output, dtype=dtype))
		print("Klarte å opprette!")


	def loss(self):
		'''
		Abstract method for loss (or cost) function:
		used by neural network to train
		supervised learning
		'''
		raise NotImplementedError

	@tf.function
	def back_propagation(self):
		'''
		back propagation in neural network
		using loss function specified by sub-class
		trainable variables : weights and bias of layers
		'''
		self.optimizer.minimize(self.loss, self.trainable_variables)


	def train(self, epoch):
		'''
		one epoch of back propagation
		'''
		for n in range(epoch):
			self.back_propagation()


	def solve(self, learning_rate, epoch, num_epochs=10, tol=1e-16):
		self.optimizer = Adam(learning_rate=learning_rate)
		for n in range(num_epochs):
			self.train(epoch)
			loss = self.loss().numpy()
			if loss < tol:			# DE is solved
				break
			elif loss != loss:		# loss in nan -> values in solution will be nan-values
				print("WARNING: Neural Network failed so solve")
				break
		return loss


	def solve_verbose(self, learning_rate, epoch, num_epochs=10, tol=1e-16):
		'''
		talkative solving of differential equation
		'''
		self.optimizer = Adam(learning_rate=learning_rate)
		for n in range(num_epochs):
			start_time = time()
			self.train(epoch)
			loss = self.loss().numpy()
			print(f"{loss:10.3e} \t {float(time() - start_time):20.5f}")
			if loss < tol:			# DE is solved
				break
			elif loss != loss:		# loss in nan -> values in solution will be nan-values
				print("WARNING: Neural Network failed so solve")
				break
		return loss


	def array_to_tensor(self, array):
		'''
		convert 1D-array to tensor
		'''
		return tf.reshape(tf.convert_to_tensor(array), shape=(-1, 1))
