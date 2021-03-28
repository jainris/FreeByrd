import numpy as np
from numpy.linalg import norm
from sklearn.metrics import pairwise as pw
from scipy.spatial import distance

# Following class is adapted from Hibrja, Vascon, Stadelmann and Pelillo (Speaker Clustering
# Using Dominant Sets) with code at https://github.com/feliksh/SCDS
class DominantSetClustering:
    """
    Dominant Set Clustering algorithm and associated methods.

    Attributes
    ----------
    feature_vectors : np.array of float, shape [num_of_inputs, num_of_features]
        The feature vectors of the inputs.

    metric : string
        {'euclidean' or 'cosine'}
        If set to 'euclidean', then euclidean distances are ussed for calculating the similarity
        matrix. Else, cosine distances are used.

    relative_cutoff : bool
        If true, relative cutoff is used.
        Else, absolute cutoff is used.

    cutoff : float
        The cutoff value to be used for clusters.
        If relative off is being used, then absolute cutoff value = cutoff * max(participating value).

    sim_matrix : np.array of float, shape [num_of_inputs, num_of_inputs]
        The similarity matrix of inputs.

    num_of_inputs : int
        Number of inputs, which is determined from the 1st dimension of feature_vectors.

    mutable_element_ids : np.array of int, shape [num_of_inputs]
        Maps the idx used in update_cluster to the actual indices of elements.
        Intially is just <1, 2, 3, 4, ....., num_of_inputs>

    cluster_ids : np.array of int, shape [num_of_inputs]
        The cluster numbers of each input, as determined by dominant set clustering algorithm.

    participating_values : np.array of float, shape [num_of_inputs]
        The participating values of inputs, as determined by dominant set clustering algorithm.

    cluster_counter : int
        The number of clusters calculated.

    reassignment : string
        {'whole', 'single' or 'noise'}
        The reassignment strategy to be used for elements that couldn't be clustered.
        whole  : All remaining clusters put in a one cluster.
        single : Each remaining cluster put into single clusters of their own.
        noise  : It is assumed that these elements weren't assigned because of noise and so,
                 they're assigned to the cluster they're most similar to.
        By default set to noise.

    epsilon : float
        The cutoff epsilon to be used for Replicator Dynamics.

    Methods
    -------
    get_num_of_clusters()
        Returns the number of clusters.

    get_participating_values()
        Returns the participating values of the elements.

    apply_clustering()
        Returns the clusters after applying the clustering algorithm.
    """

    def __init__(
        self,
        feature_vectors,
        metric="cosine",
        relative_cutoff=True,
        cutoff=0.1,
        reassignment="noise",
        epsilon=1e-6,
    ):
        """
        Intializes the object.

        Parameters
        ----------
        feature_vectors : np.array of float, shape [num_of_inputs, num_of_features]
            The feature vectors of the inputs.

        metric : string, optional
            {'euclidean' or 'cosine'}
            If set to 'euclidean', then euclidean distances are ussed for calculating the similarity
            matrix. Else, cosine distances are used.
            By default set to 'cosine'.

        relative_cutoff : bool, optional
            If true, relative cutoff is used.
            Else, absolute cutoff is used.
            By default set to True.

        cutoff : float, optional
            The cutoff value to be used for clusters.
            If relative off is being used, then absolute cutoff value = cutoff * max(participating value).
            By default set to 0.1.

        reassignment : string, optional
            {'whole', 'single' or 'noise'}
            The reassignment strategy to be used for elements that couldn't be clustered.
            whole  : All remaining clusters put in a one cluster.
            single : Each remaining cluster put into single clusters of their own.
            noise  : It is assumed that these elements weren't assigned because of noise and so,
                     they're assigned to the cluster they're most similar to.
            By default set to noise.

        epsilon : float, optional
            The cutoff epsilon to be used for Replicator Dynamics.
            By default set to 1e-6.
        """
        self.feature_vectors = np.array(feature_vectors, dtype=np.float)
        self.metric = metric
        self.relative_cutoff = relative_cutoff
        self.cutoff = cutoff
        self.sim_matrix = None
        self.num_of_inputs = feature_vectors.shape[0]
        self.mutable_element_ids = np.array(range(self.num_of_inputs))
        self.cluster_ids = np.zeros(shape=self.num_of_inputs, dtype=int) - 1
        self.participating_values = np.zeros(shape=self.num_of_inputs)
        self.cluster_counter = 0
        self.reassignment = reassignment
        self.epsilon = epsilon

    def _set_cluster(self, idx, values):
        """
        Sets the cluster id of elements mapping to idx and increments the cluster id by 1.
        Also sets the values of these elements to be their participating values
        in the characteristic vector.

        Parameters
        ----------
        idx : np.array of bool, shape [N]
            Vector with value of an element set to true if it is supposed to be in the cluster,
            else false.

        values : np.array of float, shape [N]
            The Characterisitic Vector.

        Returns
        -------
        None
        """
        # in results vector (self.cluster_ids) assign cluster number to elements of idx
        # values: are participating values of charecteristic vector of DS
        self.cluster_ids[self.mutable_element_ids[idx]] = self.cluster_counter
        self.participating_values[self.mutable_element_ids[idx]] = values[idx]
        self.mutable_element_ids = self.mutable_element_ids[idx == False]
        self.cluster_counter += 1

    def get_num_of_clusters(self):
        """ Returns the number of clusters. """
        return self.cluster_counter

    def get_participating_values(self):
        """ Returns the participating values of the elements. """
        return self.participating_values

    def _get_sim_matrix(self, feature_vectors=None, num_of_nearest_neighbours=7):
        """
        Returns and sets the similarity matrix.
        (High value reflects highly similar)

        Parameters
        ----------
        feature_vectors : np.array of float, shape [num_of_inputs, num_of_features], optional
            The feature vectors. If None, then set to the feature vectors the object is
            initialized with.

        num_of_nearest_neighbours : int, optional
            Number of nearest neighbours to be used for calculating the sigma_i of a node i.
            By default set to 7.

        Returns
        -------
        sim_matrix: np.array with type = float and shape = [num_of_inputs, num_of_inputs]
            The similarity matrix calculated using euclidean distances or cosine distances.
        """
        if feature_vectors is None:
            feature_vectors = self.feature_vectors
        if self.metric == "euclidean":
            dist_mat = distance.pdist(feature_vectors, metric=self.metric)
            dist_mat = distance.squareform(dist_mat)
        else:
            # using cosine distance
            dist_mat = pw.cosine_similarity(feature_vectors)

            # accounting for floating pt. error which can lead to
            # the cosine value not being in the [-1, 1] range
            dist_mat = np.where(dist_mat > 1, 1, dist_mat)
            dist_mat = np.where(dist_mat < -1, -1, dist_mat)

            dist_mat = np.arccos(dist_mat)
            # ensuring that the diagonal is 0
            dist_mat[np.eye(dist_mat.shape[0]) > 0] = 0
            dist_mat /= np.pi

        # the following heuristic is derived from Perona 2005 (Self-tuning spectral clustering)
        # with adaption from Zemene and Pelillo 2016 (Interactive image segmentation using
        # constrained dominant sets)

        # sigma[i, j] = sigma_i * sigma_j
        #       ,where sigma_i is calculated as below

        # sigma_i = sum(d(f_i, f_k))/num_of_nearest_neighbours
        #       ,where f_i is the feature vector of node i,
        #              f_k is the feature vector of node k,
        #              k is local neighbouring node

        # calculating the nearest neighbours (more similar neighbours are said to be nearer)
        # not taking the 1st element, as that would be the node itself
        sigmas = np.sort(dist_mat, axis=1)[:, 1 : (num_of_nearest_neighbours + 1)]
        # calculating sigma_i
        sigmas = np.mean(sigmas, axis=1)
        # calculating sigma
        sigmas = np.dot(sigmas[:, np.newaxis], sigmas[np.newaxis, :])

        # sim_matrix[i, j] = exp(-d(f_i, f_j)/sigma[i, j])
        # Need to take care if sigma[i, j] == 0
        ids = sigmas == 0
        sigmas[ids] = 1
        dist_mat /= -sigmas
        sim_matrix = np.exp(dist_mat)
        sim_matrix[ids] = 0

        # zeros in main diagonal needed for dominant sets
        sim_matrix = sim_matrix * (1.0 - np.identity(sim_matrix.shape[0]))
        return sim_matrix

    def _reassign(self, x):
        """
        Assigns any left out elements to clusters based on set preference.

        Parameters
        ----------
        x : np.array of float, shape [N]
            The characteristic vector
        """
        if self.reassignment == "whole":
            self._set_cluster(np.asarray(x) >= 0.0, np.zeros(shape=len(x)))
        elif self.reassignment == "single":
            x = np.ones(x.shape[0])
            while np.count_nonzero(x) > 0:
                temp = np.zeros(shape=x.shape[0], dtype=bool)
                # setting only the first element to be put in a new cluster
                temp[0] = True
                self._set_cluster(temp, np.zeros(shape=x.shape[0]))
                x = x[1:]
        else:
            # noise reassignment
            temp_sim = self.sim_matrix
            for id in range(0, x.shape[0]):
                # getting the feature vector of the id'th element not assigned
                features = self.feature_vectors[self.mutable_element_ids[id]]
                # initializing nearest and cluster_id
                nearest = 0.0
                cluster_id = 0
                # comparing id'th element's features with each cluster
                for i in range(0, self.cluster_counter):
                    cluster_elements = self.feature_vectors[self.cluster_ids == i]
                    if len(cluster_elements) > 0:
                        # finding the dominant element of the cluster using the fact that
                        # the dominant element would have the highest participating value
                        cluster_vls = self.participating_values[self.cluster_ids == i]
                        dominant_element = cluster_elements[
                            cluster_vls == np.max(cluster_vls)
                        ][0]
                        feature_vectors = np.asmatrix([features, dominant_element])
                        # getting the similarity matrix for the selected 2 elements
                        similarity = self._get_sim_matrix(
                            feature_vectors=feature_vectors
                        )[0, 1]

                        if similarity > nearest:
                            cluster_id = i
                            nearest = similarity
                self.cluster_ids[self.mutable_element_ids[id]] = cluster_id
                self.participating_values[self.mutable_element_ids[id]] = 0.0

    def apply_clustering(self):
        """
        Returns the clusters after applying the clustering algorithm.

        Returns
        -------
        cluster_ids : np.array of int, shape [N]
            The cluster_ids as determined by the clustering algorithm.
        """
        # TODO: Once rest of the model is implemented (Bulbul and BirdNet stuff), check whether
        # we get better results with current initialization of the characterisitic vector or
        # when it is initialized uniformly (which I've commented currently)
        np.random.seed(0)
        # calculating similarity matrix based on metric
        self.sim_matrix = self._get_sim_matrix()

        A = self.sim_matrix
        # initializing x (characterisitic vector)
        # x = np.ones(self.num_of_inputs) / float(self.num_of_inputs)
        x = np.random.rand((self.num_of_inputs))
        x = x / np.sum(x)
        # repeating until all objects have been clustered
        while x.size > 1:
            # using replicator dynamics:
            dist = self.epsilon * 2
            # repeating until convergence (dist < epsilon means convergence)
            while dist > self.epsilon and A.sum() > 0:
                x_old = x.copy()
                # applying replicator dynamics
                x = x * A.dot(x)
                x = x / x.sum() if x.sum() > 0.0 else x
                # calculating distance
                dist = norm(x - x_old)

            temp_cutoff = self.cutoff
            if self.relative_cutoff:
                temp_cutoff = self.cutoff * np.max(x)

            # in case of elements not belonging to any cluster at the end,
            # we assign each of them based on self.reassignment preference
            if A.sum() == 0 or sum(x >= temp_cutoff) == 0:
                print("Using reassignment for " + str(x.size) + " elements.")
                self._reassign(x)
                return self.cluster_ids

            # those elements whose value is >= temp_cutoff are the ones belonging to the cluster just found
            # on x are their participating values (characteristic vector)
            self._set_cluster(x >= temp_cutoff, x)

            # finding the elements which are left
            idx = x < temp_cutoff
            # peeling off the dominant set
            A = A[idx, :][:, idx]
            # re-initializing the characterisitic vector
            # x = np.ones(A.shape[0]) / float(A.shape[0])
            x = np.random.rand((A.shape[0]))
            x = x / np.sum(x)
        # if one element remains, putting it in a cluster of itself
        if x.size > 0:
            self._set_cluster(x >= 0.0, x)

        return self.cluster_ids


def get_clusters_using_dominant_sets(
    inputs,
    metric="cosine",
    relative_cutoff=True,
    cutoff=0.1,
    reassignment="noise",
    epsilon=1e-6,
):
    """
    Returns the cluster ids as obtained from dominant sets clustering algorithm on the given inputs.

    Parameters
    ----------
    inputs : np.array of float
        The inputs, with the first dimension as the number of inputs.

    metric : string, optional
        {'euclidean' or 'cosine'}
        If set to 'euclidean', then euclidean distances are ussed for calculating the similarity
        matrix. Else, cosine distances are used.
        By default set to 'cosine'.

    relative_cutoff : bool, optional
        If true, relative cutoff is used.
        Else, absolute cutoff is used.
        By default set to True.

    cutoff : float, optional
        The cutoff value to be used for clusters.
        If relative off is being used, then absolute cutoff value = cutoff * max(participating value).
        By default set to 0.1.

    reassignment : string, optional
        {'whole', 'single' or 'noise'}
        The reassignment strategy to be used for elements that couldn't be clustered.
        whole  : All remaining clusters put in a one cluster.
        single : Each remaining cluster put into single clusters of their own.
        noise  : It is assumed that these elements weren't assigned because of noise and so,
                    they're assigned to the cluster they're most similar to.
        By default set to noise.

    epsilon : float, optional
        The cutoff epsilon to be used for Replicator Dynamics.
        By default set to 1e-6.

    Returns
    -------
    cluster_ids : np.array of int, shape [N]
        The cluster_ids as determined by the clustering algorithm.
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
    dsc = DominantSetClustering(
        inputs, metric, relative_cutoff, cutoff, reassignment, epsilon
    )
    # applying the clustering
    cluster_ids = dsc.apply_clustering()
    return cluster_ids
