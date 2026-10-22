import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	
	scores = np.asarray(scores)
	c = scores.max()

	return scores - c - np.log(np.exp(scores - c).sum())


