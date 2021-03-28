import numpy as np
import random
import pytest

import ML.Clusterer.dominantset as ds


def generate_feature_vectors_from_cluster_ids(cluster_ids):
    """
    Utility function. Generate feature vectors from cluster ids by using one hot encoding.
    One Hot encoding is being used because for small values it should be easy for the algorithm
    to differentiate the clusters. Note that this won't be true for larger number of values or
    for some special cases. However, for small values, we can use it to see if that the clustering
    is working.

    Parameters
    ----------
    cluster_ids : np.array of int, shape [N]
        The cluster ids. Assumes that cluster ids are non-negative.

    Returns
    -------
    feature_vectors : np.array of int, shape [N, M]
        The feature vectors using one-hot encoding.
    """
    cluster_ids = np.array(cluster_ids)
    num_of_clusters = np.max(cluster_ids) + 1
    feature_vectors = np.zeros((cluster_ids.shape[0], num_of_clusters))
    for i in range(cluster_ids.shape[0]):
        feature_vectors[i, cluster_ids[i]] = 1
    return feature_vectors


def generate_feature_vectors_and_cluster_ids(num_of_inputs, max_num_of_clusters=None):
    """
    Utility function. Randomly generates cluster_ids and featuer vectors (using one-hot encoding)
    with given number of inputs.
    Note: There is a further constraint put on the cluster ids, that there can't be 2 different
    clusters with only one element each. This is because we are using one-hot encoding as feature
    vectors and so the algorithm won't be able to tell that these are supposed to be different
    clusters, and would group them together.

    Parameters
    ----------
    num_of_inputs : int
        The number of inputs.

    max_num_of_clusters : int, optional
        The maximum number of clusters. If None or out of bounds, then max is taken to be
        ceil(num_of_inputs/2).

    Returns
    -------
    cluster_ids : np.array of int, shape [N]

    feature_vectors : np.array of float, shape [N, M]
    """
    if (
        max_num_of_clusters is None
        or max_num_of_clusters <= 0
        or max_num_of_clusters > int((num_of_inputs + 1) / 2)
    ):
        max_num_of_clusters = int((num_of_inputs + 1) / 2)
    cluster_ids = np.zeros(num_of_inputs)
    cluster_ids = np.random.randint(max_num_of_clusters, size=num_of_inputs)
    # ensuring than there is atmost one single element cluster (or else the algorithm won't be
    # able to differentiate between the single element clusters and would group them together)
    count = 2
    while count > 1:
        count = 0
        vals = np.unique(cluster_ids)
        for id in vals:
            idx = cluster_ids == id
            if np.sum(idx) == 1:
                count += 1
                if count > 1:
                    cluster_ids[idx] = random.choice(vals)
                    break

    feature_vectors = generate_feature_vectors_from_cluster_ids(cluster_ids)

    return cluster_ids, feature_vectors


def compare_clusters(expected_cluster_ids, actual_cluster_ids):
    """
    Utility function. Compares expected cluster ids with actual cluster ids.
    Returns True if they match, else False.
    This is needed because the clusters might be equivalent even though the cluster ids
    might be different.

    Parameters
    ----------
    expected_cluster_ids : np.array of int, shape [N]

    actual_cluster_ids : np.array of int, shape [N]

    Returns
    -------
    answer : boolean
        True if both match.
        Else False.
    """
    expected_cluster_ids = np.array(expected_cluster_ids)
    actual_cluster_ids = np.array(actual_cluster_ids)

    # Some assertions to make debugging easier
    assert (
        len(actual_cluster_ids.shape) == 1
    ), "Unexpected shape {} of actual_cluster_ids".format(actual_cluster_ids.shape)
    assert (
        len(expected_cluster_ids.shape) == 1
    ), "Unexpected shape {} of expected_cluster_ids".format(expected_cluster_ids.shape)
    assert (
        actual_cluster_ids.shape[0] == expected_cluster_ids.shape[0]
    ), "The number of inputs should be the same."

    act_to_exp_cids = dict()
    exp_ids_seen = set()

    for i in range(expected_cluster_ids.shape[0]):
        if expected_cluster_ids[i] in exp_ids_seen:
            if (
                actual_cluster_ids[i] not in act_to_exp_cids.keys()
                or act_to_exp_cids[actual_cluster_ids[i]] != expected_cluster_ids[i]
            ):
                return False
        else:
            exp_ids_seen.add(expected_cluster_ids[i])
            if actual_cluster_ids[i] in act_to_exp_cids.keys():
                return False
            act_to_exp_cids[actual_cluster_ids[i]] = expected_cluster_ids[i]

    return True


def test_clustering_with_upto_10_inputs():
    def test_clustering_with_n_inputs(n):
        (
            expected_cluster_ids,
            feature_vectors,
        ) = generate_feature_vectors_and_cluster_ids(num_of_inputs=n)
        actual_cluster_ids = ds.get_clusters_using_dominant_sets(feature_vectors)
        assert compare_clusters(
            expected_cluster_ids, actual_cluster_ids
        ), "Failed for number of inputs = {}.\nExpected: {}\nActual: {}".format(
            n, expected_cluster_ids, actual_cluster_ids
        )

    np.random.seed(0)
    for n in range(1, 11):
        test_clustering_with_n_inputs(n)
