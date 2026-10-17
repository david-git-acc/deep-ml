import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	
	M = np.array(matrix)
	b = -np.trace(M)
	c = np.linalg.det(M)

	roots = np.roots([1, b, c])
	return roots.tolist()