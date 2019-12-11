import numpy as np 
from eigen_problem import EigenProblem 
import tensorflow as tf

def test(A):
	N = 2
	NN = EigenProblem(A, [7]*N, ["sigmoid"]*N)
	eta = 0.05
	epoch = 10000
	loss = NN.solve_verbose(eta, epoch, num_epochs=10, tol=np.nan)
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


#print(np.random.randint(low=0, high=1000000, size=100)/3.14)

n = 6
for i in range(10):
	A = create_symmetric_matrix(n)
	test(A)
	print()
