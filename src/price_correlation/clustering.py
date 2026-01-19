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
    Find optimal cluster count using silhouette score with elbow detection.

    Uses a combination of silhouette score and elbow detection to avoid
    selecting trivially small cluster counts (like k=2) when the silhouette
    score plateau is nearly flat.

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

    # For large datasets, set a reasonable minimum to avoid trivial clustering
    if n_samples > 100:
        min_k = max(min_k, 5)  # At least 5 clusters for 100+ stocks
    if n_samples > 500:
        min_k = max(min_k, 10)  # At least 10 clusters for 500+ stocks
    if n_samples > 1000:
        min_k = max(min_k, 15)  # At least 15 clusters for 1000+ stocks

    print(f"  Finding optimal k: n_samples={n_samples}, min_k={min_k}, max_k={max_k}", flush=True)

    # For large datasets, evaluate fewer k values to speed up
    if n_samples > 500:
        # Sample k values instead of trying every one
        k_values = list(range(min_k, min(30, max_k + 1)))  # Check min_k to 30
        k_values.extend(range(30, max_k + 1, 5))  # Then sample every 5
        k_values = sorted(set(k_values))
    else:
        k_values = list(range(min_k, max_k + 1))

    scores = {}
    best_k = min_k
    best_score = -1.0

    for k in k_values:
        labels = fcluster(Z, t=k, criterion="maxclust")
        n_clusters = len(set(labels))

        if n_clusters < 2:
            continue

        try:
            score = silhouette_score(distance_matrix, labels, metric="precomputed")
            scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k
        except ValueError:
            continue

    # Log top scores for debugging
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top silhouette scores: {[(k, f'{s:.4f}') for k, s in sorted_scores]}", flush=True)

    # If best_k is suspiciously low and scores are similar, use elbow method
    if best_k < 10 and len(scores) > 10:
        # Check if there's a better k with similar score (within 10%)
        threshold = best_score * 0.90
        candidates = [(k, s) for k, s in scores.items() if s >= threshold and k >= 10]
        if candidates:
            # Choose the one with highest score among larger k values
            alt_k, alt_score = max(candidates, key=lambda x: x[1])
            print(f"  Adjusting from k={best_k} to k={alt_k} (scores similar: {best_score:.4f} vs {alt_score:.4f})", flush=True)
            best_k = alt_k
            best_score = alt_score

    print(f"  Selected k={best_k} with silhouette={best_score:.4f}", flush=True)
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
