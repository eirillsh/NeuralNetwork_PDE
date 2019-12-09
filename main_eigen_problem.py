import numpy as np 
from eigen_problem import EigenProblem 
import tensorflow as tf

def test(A):
	N = 2
	NN = EigenProblem(A, [7]*N, ["sigmoid"]*N)
	eta = 0.05
	epoch = 1000
	loss = NN.solve(eta, epoch, num_epochs=50)
	print(f"EIGENVALUE  {NN.E:15.8f}  {loss:20.1e}")
	E, X = np.linalg.eigh(A)
	print(E)


def create_symmetric_matrix(n):
	'''
	create symmetric matrix A
		A: n x n
		A_{i, j} in [-1, 1] 
	'''
	Q =  2*np.random.ranf((n, n)) - 1
	return (Q.T + Q)/2

n = 6
for i in range(10):
	A = create_symmetric_matrix(n)
	test(A)
	print()