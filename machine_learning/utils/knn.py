import numpy as np

class KNearestNeighbor(object):
	""" a kNN classifier with L2 distance """

	def __init__(self):
		pass

	def train(self, X, y):
		"""
		Train the classifier. For k-nearest neighbors this is just
		memorizing the training data.

		Inputs:
		- X: A numpy array of shape (num_train, D) containing the training data
		consisting of num_train samples each of dimension D.
		- y: A numpy array of shape (N,) containing the training labels, where
			y[i] is the label for X[i].
		"""
		self.X_train = X
		self.y_train = y

	def predict(self, X, k=1):
		"""
		Predict labels for test data using this classifier.

		Inputs:
		- X: A numpy array of shape (num_test, D) containing test data consisting
				of num_test samples each of dimension D.
		- k: The number of nearest neighbors that vote for the predicted labels.
		- method: Determines which implementation to use to compute distances
			between training points and testing points.

		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
			test data, where y[i] is the predicted label for the test point X[i].
		"""
		dists = self.compute_distances_l2norm(X)
		return self.predict_labels(dists, k=k)

	def compute_distances_l2norm(self, X):
		"""
		Compute the distance between each test point in X and each training point
		in self.X_train using no explicit loops.

		Input / Output: Same as compute_distances_two_loops
		"""
		num_test = X.shape[0]
		num_train = self.X_train.shape[0]
		dists = np.zeros((num_test, num_train))
		# expand equation (x-y)^2 = x^2 + y^2 - 2xy
		dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(self.X_train **
																2, axis=1) - 2 * np.matmul(X, self.X_train.T)
		dists = np.sqrt(dists) # (13854, 6)

		return dists

	def predict_labels(self, dists, k=1):
		"""
		Given a matrix of distances between test points and training points,
		predict a label for each test point.

		Inputs:
		- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
			gives the distance betwen the ith test point and the jth training point.

		Returns:
		- y: A numpy array of shape (num_test,) containing predicted labels for the
			test data, where y[i] is the predicted label for the test point X[i].  
		"""
		num_test = dists.shape[1]

		closest_y = []
		for i in range(num_test):
			closest_y.append(np.argsort(dists[:, i])[0:k].tolist())

		return closest_y
