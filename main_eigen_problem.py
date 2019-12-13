import numpy as np
from eigen_problem import EigenProblem
import tensorflow as tf
from tensorflow.keras import activations as act
from tensorflow.keras import layers

def test(A):
	N = 2
	NN = EigenProblem(A, [10]*N, ["sigmoid"]*N)
	eta = 0.05
	epoch = 3000
	loss = NN.solve(eta, epoch, num_epochs=10)
	#print(f"EIGENVALUE  {NN.E:15.8f}  {loss:20.1e}")
	#E, X = np.linalg.eigh(A)
	#print(E)

	return NN.E.numpy()


def create_symmetric_matrix(n):
	'''
	create symmetric matrix A
		A: n x n
		A_{i, j} in [-1, 1]
	'''
	Q =  2*np.random.ranf((n, n)) - 1
	return (Q.T + Q)/2

n = 6
A = create_symmetric_matrix(n)
eigs = []
eigs.append(test(A))
print("Found eigenvalue number 1 of", n)
i = 2


while True:
	E = test(A)
	if len(eigs) == 6:
		break
	elif np.any(np.abs(eigs - E) < 1e-4):
		pass
	else:
		print("Found eigenvalue number", i, "of", n)
		eigs.append(E)
		i += 1

print(np.sort(eigs))
E, X = np.linalg.eigh(A)
print(np.sort(E))
