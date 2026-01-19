"""Clustering engine - DBSCAN and hierarchical clustering."""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def cluster_dbscan(
    distance_matrix: np.ndarray,
    eps: float,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Cluster using DBSCAN with precomputed distances.

    Args:
        distance_matrix: NxN distance matrix
        eps: Maximum distance for neighborhood
        min_samples: Minimum points to form cluster

    Returns:
        Labels array (-1 for noise)
    """
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    )
    return dbscan.fit_predict(distance_matrix)


def find_optimal_eps(
    distance_matrix: np.ndarray,
    k: int = 5,
) -> float:
    """
    Find optimal DBSCAN epsilon using k-distance graph elbow method.

    Args:
        distance_matrix: NxN distance matrix
        k: k-th nearest neighbor to consider

    Returns:
        Suggested epsilon value
    """
    nn = NearestNeighbors(n_neighbors=k, metric="precomputed")
    nn.fit(distance_matrix)
    distances, _ = nn.kneighbors(distance_matrix)

    k_distances = np.sort(distances[:, k - 1])[::-1]

    diffs = np.diff(k_distances)
    elbow_idx = np.argmax(np.abs(diffs)) + 1

    return float(k_distances[elbow_idx])


def cluster_hierarchical(
    condensed_distance: np.ndarray,
    method: str = "average",
) -> np.ndarray:
    """
    Perform hierarchical clustering.

    Args:
        condensed_distance: Condensed distance array from pdist
        method: Linkage method (single, complete, average, ward)

    Returns:
        Linkage matrix Z
    """
    return linkage(condensed_distance, method=method)


def cut_dendrogram(
    Z: np.ndarray,
    n_clusters: int | None = None,
    threshold: float | None = None,
) -> np.ndarray:
    """
    Cut dendrogram to get flat cluster labels.

    Args:
        Z: Linkage matrix
        n_clusters: Desired number of clusters
        threshold: Distance threshold for cutting

    Returns:
        Cluster labels (1-indexed)
    """
    if n_clusters is not None:
        return fcluster(Z, t=n_clusters, criterion="maxclust")
    elif threshold is not None:
        return fcluster(Z, t=threshold, criterion="distance")
    else:
        raise ValueError("Must specify n_clusters or threshold")


def find_optimal_k(
    Z: np.ndarray,
    distance_matrix: np.ndarray,
    max_k: int | None = None,
    min_k: int = 2,
) -> tuple[int, float]:
    """
    Find optimal cluster count using silhouette score.

    Args:
        Z: Linkage matrix from hierarchical clustering
        distance_matrix: NxN distance matrix
        max_k: Maximum clusters to try (default: min(100, sqrt(n)*2))
        min_k: Minimum clusters to try

    Returns:
        (best_k, best_score)
    """
    n_samples = distance_matrix.shape[0]

    # Scale max_k based on dataset size
    if max_k is None:
        # Use sqrt(n) * 2 as a reasonable upper bound, capped at 100
        max_k = min(100, max(30, int(np.sqrt(n_samples) * 2)))

    best_k = min_k
    best_score = -1.0

    # For large datasets, evaluate fewer k values to speed up
    if n_samples > 500:
        # Sample k values instead of trying every one
        k_values = list(range(min_k, min(20, max_k + 1)))  # Always check 2-20
        k_values.extend(range(20, max_k + 1, 5))  # Then sample every 5
        k_values = sorted(set(k_values))
    else:
        k_values = list(range(min_k, max_k + 1))

    for k in k_values:
        labels = fcluster(Z, t=k, criterion="maxclust")
        n_clusters = len(set(labels))

        if n_clusters < 2:
            continue

        try:
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_k = k
        except ValueError:
            continue

    return best_k, best_score


def auto_cluster(
    distance_matrix: np.ndarray,
    condensed_distance: np.ndarray,
    method: str = "hierarchical",
) -> tuple[np.ndarray, dict]:
    """
    Automatic clustering with parameter selection.

    Returns:
        (labels, metadata_dict)
    """
    if method == "dbscan":
        eps = find_optimal_eps(distance_matrix)
        labels = cluster_dbscan(distance_matrix, eps=eps)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        return labels, {
            "method": "dbscan",
            "eps": eps,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }

    else:
        Z = cluster_hierarchical(condensed_distance, method="average")
        best_k, best_score = find_optimal_k(Z, distance_matrix)
        labels = cut_dendrogram(Z, n_clusters=best_k)
        labels = labels - 1

        return labels, {
            "method": "hierarchical",
            "n_clusters": best_k,
            "silhouette": best_score,
            "linkage": "average",
        }
