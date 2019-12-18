import numpy as np
from eigen_problem import EigenProblem
import tensorflow as tf
from tensorflow.keras import activations as act
from tensorflow.keras import layers
from time import time
from tabulate import tabulate
import matplotlib.pyplot as plt
from tqdm import tqdm

path = "Figures/"
fsize = 15

def test(A):
	N = 2
	NN = EigenProblem(A, [10]*N, ["sigmoid"]*N)
	eta = 0.1
	epoch = 4000
	loss = NN.solve(eta, epoch, num_epochs=10, tol=1e-18)
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
	return (Q + Q.T)/2
	
run = False

if run:
	n = 6
	A = create_symmetric_matrix(n)
	A = np.around(A, decimals=4)

	alleigs = []


	for i in tqdm(range(30)):
		alleigs.append(np.around(test(A), decimals=4))

	E, X = np.linalg.eigh(A)
	E = np.sort(np.around(E, decimals=4))
	alleigs = np.sort(alleigs)
	print(E)
	print(tabulate(A, tablefmt="latex", floatfmt=".4f"))

	count = np.zeros(n)
	for i, eig in enumerate(E):
		k = 0
		for alleig in alleigs:
			if np.abs(eig - alleig) < 1e-2:
				k += 1
		count[i] = k
	xind = np.arange(1, len(count) + 1)
	yind = np.arange(0, np.max(count) + 1)
	plt.bar(xind, count, width=0.3)
	plt.xticks(xind, (r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$", r"$\lambda_5$", r"$\lambda_6$"), fontsize=fsize)
	plt.yticks(yind, fontsize=fsize)
	plt.xlabel("Eigenvalues", fontsize=fsize)
	plt.tight_layout()
	plt.savefig(path + "eigs.pdf")
	plt.show()
