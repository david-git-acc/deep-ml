import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	
	X = np.asarray(X)
	y = np.asarray(y)
	ones = np.ones(shape = (len(X), 1))

	feature_matrix = np.hstack((ones, X))
	w = np.linalg.inv(X.T @ X) @ (X.T @ y)
	w = np.round(w, decimals = 4)
	return w