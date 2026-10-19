import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
	
	A = np.asarray(A)
	S = np.asarray(S)
	T = np.asarray(T)

	both_invertible = np.linalg.det(T) != 0 and np.linalg.det(S) != 0

	if both_invertible:
		return np.linalg.inv(T) @ A @ S
	return -1