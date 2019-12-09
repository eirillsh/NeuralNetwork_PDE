import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer 
from tensorflow.keras.optimizers import SGD as GradientDecent
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
	
	# create layers for feed forward neural network
	def add_feed_forward_layers(self, features, size_output, num_neurons, activation_functions):
		'''
		create layers for feed forward neural network
		'''
		dtype = 'float64'					#default is float32
		# adding input layer
		self.add(InputLayer(input_shape=(features,), dtype=dtype))
		# adding hidden layers
		for neurons, f in zip(num_neurons, activation_functions):
			self.add(Dense(neurons, activation=f, dtype=dtype))
		# adding output layer
		self.add(Dense(size_output, dtype=dtype))

	# create layers for recurrent neural network
	def add_recurrent_layers(self):
		''' 		ANDERS, LES MEG
		Vi kan se om vi kan prøve å få til dette. 
		Jeg har prøvd litt, men fikk ikke helt til...
		tensorflow har recurrent layers som vi burde kunne få til å bruke
		'''
		pass

	@tf.function
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

	# one epoch of back propagation
	def train(self, epoch):
		for n in range(epoch):
			self.back_propagation()

	# solve differential equation
	def solve(self, learning_rate, epoch, num_epochs=10, tol=1e-16):
		self.optimizer = GradientDecent(learning_rate=learning_rate)
		for n in range(num_epochs):
			self.train(epoch)
			loss = self.loss().numpy()
			if loss < tol:			# DE is solved
				break
			elif loss != loss:		# loss in nan -> values in solution will be nan-values
				print("WARNING: Neural Network failed so solve")
				break
		return loss

	# talkative solving of differential equation
	def solve_verbose(self, learning_rate, epoch, num_epochs=10, tol=1e-16):
		self.optimizer = GradientDecent(learning_rate=learning_rate)
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

	# Convert 1D-array to tensor 
	def array_to_tensor(self, array):
		return tf.reshape(tf.convert_to_tensor(array), shape=(-1, 1))



