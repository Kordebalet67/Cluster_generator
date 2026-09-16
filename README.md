# Cluster_generator
This code generate dataset to cluster it with some newral networks

Input parametres:
    - n_clusters (int): number of clusters(centres).
    - n_features (int): number of characteres for each element of dataset.
    - n_samples (int): Whole number of elements in dataset (default = 500).
    - random_state (int): random state of generator.

Output:
    - X (numpy.ndarray): dataset.
    - y (numpy.ndarray): label of cluster for each element.
