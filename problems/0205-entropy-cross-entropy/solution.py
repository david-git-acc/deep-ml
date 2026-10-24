import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""

	P = np.asarray(P)
	Q = np.asarray(Q)

	# Your code here
	entropy = cel(P, P)
	cross_entropy = cel(P, Q)
	return (entropy, cross_entropy)

def cel(P : np.ndarray, Q : np.ndarray) -> float:

	terms = np.zeros(len(P))

	P_is_zero = P == 0
	Q_is_zero = Q == 0
	
	terms[(~P_is_zero) & Q_is_zero] = -float("inf")

	rest = (~P_is_zero) & (~Q_is_zero)
	terms[rest] = (P[rest] * np.log(Q[rest])).astype(float)

	return -terms.sum() 



