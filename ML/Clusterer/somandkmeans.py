import numpy as np
from sklearn.cluster import KMeans

# TODO: Try different values for dec_const.
# Following classes are inspired from Jiang and Liu (An Improved Speech Segmentation and Clustering
# Algorithm Based on SOM and K-Means)
class SOM:
    """
    Self-organisizing Maps (SOM) implementation.

    Attributes
    ----------
    feature_vectors : np.array of float, shape = [num_of_inputs, num_of_features]
        The feature vectors of the inputs.

    num_of_inputs : int
        The number of inputs vectors.

    num_of_features : int
        The number of features of each input vector.

    k : int
        The number of nodes in the competition layer.

    weight_vectors : np.array of float, shape [k, num_of_features]
        The weight vectors for each neuron.

    max_iters : int
        The maximum number of iterations in the training phase.

    grid_length : int
        The length of the grid of nodes in the competition layer.
        Note the nodes in the competition layer are arranged in a rectangular grid in the
        competition layer, and hence has a length and width.

    grid_width : int
        The width of the grid of nodes in the competition layer.

    neighbourhood_radius : float
        The radius for determining the dominant/superior neighbourhood.

    alpha : float
        The learning rate.

    dec_const : float
        The value of the constant for decrement of the coefficients (radius and alpha).

    num_wins : np.array of int, shape [k]
        The number of times each node has been the winner.

    min_alpha : float
        The minimum value of the learning rate, reaching which the training phase is stopped.

    rel_cutoff : float
        The value of cutoff to be used for determining the actual number of clusters.
        Note, this is not actually needed for the original SOM. However, we require this for
        our purpose of SOM and k-means hybrid.

    Methods
    -------
    applySOM()
        Runs the training phase of the SOM. Returns the nodes which have been the winner nodes
        more than average.

    Notes
    -----
    * Only the training phase is implemented as only that is required for our purposes.
    * Only the dominant neighbourhood (also known as superior neighbourhood) can adjust the
      weight vectors. (This is the basis of competitive learning, however in this implementation
      neighbourhood of the winning neuron is taken instead of just the winning neuron/node)
      To see how the dominant neighbourhood is computed, see the following points.
    * It arranges the k nodes of the competition layer as a grid as follows:
        <-length->
        O - O - O  ^
        | X | X |  |
        O - O - O width
        | X | X |  |
        O - O - O  V
           , where O represents a node and -, /, \ and | represent a direct link.
             X is just the 2 links / and \ overlapping.
      A direct link between 2 nodes mean that they are neighbours and the topological distance
      between them is 1. If a node can be reached from another node with a minimum of d direct
      links, then the nodes are said to be at a topological distance d.
    * The indices of the nodes are mapped to the 2-D grid as follows
        0  1  2
        3  4  5
        6  7  8
    * The dominant neighbourhood is said to be all the nodes within some topological radius r of
      the winning node. This r is said to be the neighbourhood radius.
    * The neighbourhood radius and learning rate decrease with each iteration.
    * The training phase stops when either the max number of iterations is reached or the min
      value of the learning rate is reached.
    * The implementation finally returns the an array of bool which corresponds to the nodes which
      won more than average in the training phase. This is useful for determining the number of
      clusters for our given data.
    """

    def __init__(
        self,
        feature_vectors,
        num_of_inputs,
        num_of_features,
        k=9,
        max_iters=100,
        initial_neighbourhood_radius=2,
        initial_alpha=1.0,
        dec_const=1.0,
        grid_length=3,
        initial_weights=None,
        min_alpha=1e-6,
        rel_cutoff=4 / 3,
    ):
        """
        Initializes the object.

        Parameters
        ----------
        feature_vectors : np.array of float, shape = [num_of_inputs, num_of_features]
            The feature vectors of the inputs.

        num_of_inputs : int
            The number of inputs vectors.

        num_of_features : int
            The number of features of each input vector.

        k : int, optional
            The number of nodes in the competition layer.
            Note, that this is supposed to be lesser than the number of inputs and hence the
            minimum of the two is taken to be the value of k.
            By default set to 9.

        max_iters : int, optional
            The maximum number of iterations in the training phase.
            By default set to 100.

        initial_neighbourhood_radius : int, optional
            The initial value of radius for determining the dominant/superior neighbourhood.
            By default set to 2.

        initial_alpha : float, optional
            The initial value for the learning rate.
            By default set to 1.

        dec_const : float, optional
            The value of the constant for decrement of the coefficients (radius and alpha).
            By default set to 1.

        grid_length : int, optional
            The length of the grid of nodes in the competition layer. This is required because the
            nodes in the competition layer are arranged in a rectangular grid in the competition layer.
            By default set to 3.

        initial_weights : np.array of float, shape [k, num_of_features], optional
            The initial value set to the weight vectors. If None, then assigned random values selected.
            By default None.

        min_alpha : float, optional
            The minimum value of the learning rate, reaching which the training phase is stopped.
            By default set to 1e-6.

        rel_cutoff : float, optional
            The value of cutoff to be used for determining the actual number of clusters.
            Note, this is not actually needed for the original SOM. However, we require this for
            our purpose of SOM and k-means hybrid.
        """
        self.feature_vectors = np.array(feature_vectors)
        self.num_of_inputs = num_of_inputs
        self.num_of_features = num_of_features
        assert (
            self.feature_vectors.shape[0] == num_of_inputs
            and self.feature_vectors.shape[1] == num_of_features
        ), "Input dimensions do not match."
        k = min(k, self.num_of_inputs)
        self.k = k
        if initial_weights is None:
            self.weight_vectors = np.random.rand((self.num_of_features * k)).reshape(
                (k, self.num_of_features)
            )
        else:
            self.weight_vectors = initial_weights
        # Normalzing the initial weights
        self.weight_vectors = self.weight_vectors / np.repeat(
            np.sum(self.weight_vectors, axis=1), self.num_of_features
        ).reshape((k, self.num_of_features))
        self.max_iters = max_iters
        self.grid_length = min(k, grid_length)
        self.grid_width = (k + grid_length - 1) // grid_length
        self.neighbourhood_radius = min(
            initial_neighbourhood_radius, max(grid_length, self.grid_width)
        )
        self.alpha = initial_alpha
        self.dec_const = dec_const
        self.num_wins = np.zeros((k))
        self.min_alpha = min_alpha
        self.rel_cutoff = rel_cutoff

    def _calc_neighbourhood(self, i):
        """
        Returns the dominant neighbourhood.
        This is based on the topological distance.

        Paramters
        ---------
        i : int
            The winning node.

        Returns
        -------
        neighbourhood : set of int
            The dominant neighbourhood set.
        """
        neighbourhood = {i}
        x, y = i % self.grid_length, int(i // self.grid_length)
        n_s = int(self.neighbourhood_radius)
        if n_s == 0:
            return neighbourhood
        min_x, max_x = max(0, x - n_s), min(self.grid_length - 1, x + n_s)
        min_y, max_y = max(0, y - n_s), min(self.grid_width - 1, y + n_s)
        for a in range(min_x, max_x + 1):
            for b in range(min_y, max_y):
                neighbourhood.add(a + b * self.grid_length)
        if max_y == self.grid_width - 1 and self.k % self.grid_length != 0:
            # The last row is not full. Therefore, avoiding adding a non-existent node.
            max_x = self.k % self.grid_length
        # Loop for last row
        for a in range(min_x, max_x):
            neighbourhood.add(int(a + max_y * self.grid_length))
        return neighbourhood

    def _neighbourhood_function(self, winner, i):
        """
        Returns the effect of the topological distance on the learning.

        Parameters
        ----------
        winner : int
            The index of the winning node.

        i : int
            The index of the node we want to get the coefficient for.
        """
        x_w = winner % self.grid_length
        x = i % self.grid_length
        dist = abs(x - x_w)
        return np.exp(
            -dist * dist / (2 * self.neighbourhood_radius * self.neighbourhood_radius)
        )

    def _update_weights(self, winner, i):
        """
        Given the winning neuron, it calculates the dominant neighbourhood and updates the
        weight vectors for these neurons.

        Parameters
        ----------
        winner : int
            The index of the winning node/neuron.

        i : int
            The index of the input vector.
        """
        neighbourhood = self._calc_neighbourhood(winner)
        for j in neighbourhood:
            self.weight_vectors[j] = self.weight_vectors[
                j
            ] + self.alpha * self._neighbourhood_function(winner, j) * (
                self.feature_vectors[i] - self.weight_vectors[j]
            )
            # normalizing the weights
            self.weight_vectors = self.weight_vectors / np.repeat(
                np.sum(self.weight_vectors, axis=1), self.num_of_features
            ).reshape((self.k, self.num_of_features))

    def _update_coeffs(self, t):
        """
        Updates the learning rate and neighbourhood radius. These are supposed to be decreasing
        with each iteration.

        Parameters
        ----------
        t : int
            The iteration number.
        """
        # using the following formulas:
        # coeff       = exp(-t/lambda) , where lambda is a constant
        # alpha(t+1)  = alpha(t) * coeff
        # radius(t+1) = radius(t) * coeff
        coeff = np.exp(-t / self.dec_const)
        self.alpha *= coeff
        self.neighbourhood_radius *= coeff

    def applySOM(self):
        """
        Runs the training phase of the SOM. Returns the nodes which have been the winner nodes
        more than average.

        Returns
        -------
        idx : np.array of bool, shape [k]
            An element is true if corresponding node survived the cutoff based on the number of
            times it has been the winner node. Else false.
        """
        # creating a random permutation of input indices
        t = 0
        while self.alpha > self.min_alpha and t < self.max_iters:
            in_id = np.random.permutation(np.arange(self.num_of_inputs))
            for i in in_id:
                if self.alpha <= self.min_alpha or t >= self.max_iters:
                    break
                # calculating winning neuron
                winner = np.argmax(
                    np.array(self.feature_vectors[i]).dot(
                        self.weight_vectors.transpose()
                    )
                )
                self._update_weights(winner, i)
                t += 1
                self._update_coeffs(t)
                self.num_wins[winner] += 1
        cutoff = self.rel_cutoff * np.mean(self.num_wins)
        return self.num_wins > cutoff


# TODO: Check if we get better results with normalized input to k-means
class SOMAndKMeans:
    """
    Implementation of a hybrid between SOM and k-means, making use of the pros of both the
    techniques. Runs the SOM training phase twice and then the k-means algorithm.

    Attributes
    ----------
    featur_vectors : np.array of float, shape [num_of_inputs, num_of_features]
            The feature vectors.

    num_of_inputs : int
        The number of input vectors.

    num_of_features : int
        The number of features of each input vector.

    normalized_fv : np.array of float, shape [num_of_inputs, num_of_features]
        Normalized values of the feature vectors.

    num_of_clusters : int
        The number of clusters.

    Methods
    -------
    apply()
        Applies a hybrid of SOM and k-means.
        First executes a training phase of SOM with number of nodes to be max_number_of_clusters.
        Using the output of this execution, determines the actual number of clusters.
        Then, executes another traning phase of SOM with these number of clusters to obtain the
        centroids of the clusters to be used with k-means.
        Executes k-means algorithm.
    """

    def __init__(self, feature_vectors, max_num_of_clusters=9):
        """
        Initializes the object.

        Parameters
        ----------
        featur_vectors : np.array of float, shape [N, M]
            The feature vectors.

        max_num_of_clusters : int, optional
            The maximum number of clusters. Needs to just be an estimate.
            Note that the algorithm will return lesser number of clusters than the specified value.
            Hence, it is advisable to set this value to be high.
            By default set to 9. (Works well for 2, 3 or 4 actual clusters)
        """
        self.feature_vectors = np.array(feature_vectors)
        self.num_of_inputs = self.feature_vectors.shape[0]
        self.num_of_features = self.feature_vectors.shape[1]
        self.normalized_fv = self.feature_vectors / np.repeat(
            np.sum(self.feature_vectors, axis=1), self.num_of_features
        ).reshape((self.num_of_inputs, self.num_of_features))
        self.num_of_clusters = max_num_of_clusters

    def apply(self):
        """
        Applies a hybrid of SOM and k-means.
        First executes a training phase of SOM with number of nodes to be max_number_of_clusters.
        Using the output of this execution, determines the actual number of clusters.
        Then, executes another traning phase of SOM with these number of clusters to obtain the
        centroids of the clusters to be used with k-means.
        Executes k-means algorithm.

        Returns
        -------
        cluster_ids : np.array of int, shape [num_of_inputs]
            The cluster_ids as determined by the algorithm.
        """
        SOM1 = SOM(
            feature_vectors=self.normalized_fv,
            num_of_inputs=self.num_of_inputs,
            num_of_features=self.num_of_features,
            k=self.num_of_clusters,
        )
        idx = SOM1.applySOM()
        self.num_of_clusters = np.sum(idx)
        SOM2 = SOM(
            feature_vectors=self.normalized_fv,
            num_of_inputs=self.num_of_inputs,
            num_of_features=self.num_of_features,
            k=self.num_of_clusters,
        )
        SOM2.applySOM()
        # applying k-means
        cluster_ids = KMeans(
            n_clusters=self.num_of_clusters, init=SOM2.weight_vectors, n_init=1
        ).fit_predict(self.feature_vectors)
        self.num_of_clusters = np.unique(cluster_ids).shape[0]
        return cluster_ids


def normalize_cluster_ids(old_cluster_ids):
    """
    Returns cluster ids which is equivalent to the given cluster ids but is guaranteed
    to start from 0 and not have any cluster id missing.

    Parameters
    ----------
    old_cluster_ids : np.array of int, shape [N]
        The original cluster ids.

    Returns
    -------
    cluster_ids : np.array of int, shape [N]
        Equivalent normalized cluster ids.
    """
    cluster_ids = np.zeros(old_cluster_ids.shape, dtype=int)
    b = cluster_ids != 0
    i = 0
    while not np.all(b):
        x = np.argmin(b)
        x = old_cluster_ids[x]
        idx = old_cluster_ids == x
        cluster_ids[idx] = i
        i = i + 1
        b[idx] = True
    return cluster_ids


def get_clusters_using_SOM_and_k_means(inputs, max_num_of_clusters=9):
    """
    Returns the cluster ids as obtained from the SOM - k-means hybrid algorithm on the given inputs.

    Parameters
    ----------
    inputs : np.array of float
        The inputs, with the first dimension as the number of inputs.

    max_num_of_clusters : int, optional
        The maximum number of clusters. Needs to just be an estimate.
        Note that the algorithm will return lesser number of clusters than the specified value.
        Hence, it is advisable to set this value to be high.
        By default set to 9. (Works well for 2, 3 or 4 actual clusters)

    Returns
    -------
    cluster_ids : np.array of int, shape [num_of_inputs]
        The cluster_ids as determined by the algorithm.
    """
    inputs = np.array(inputs)
    if len(inputs.shape) == 0:
        # a single value, so assigning it just the first value
        return [0]
    if len(inputs.shape) == 1:
        # no feature vectors, so assuming distinct clusters for each input
        return np.arange(inputs.shape[0])
    if len(inputs.shape) > 2:
        # flattening the features of each input into feature_vectors
        new_shape = (inputs.shape[0], np.prod(inputs.shape[1:]))
        inputs = inputs.reshape(new_shape)
    # intializing the clusterer
    somk = SOMAndKMeans(inputs, max_num_of_clusters=max_num_of_clusters)
    # applying the clustering
    cluster_ids = somk.apply()
    cluster_ids = normalize_cluster_ids(cluster_ids)
    return cluster_ids
