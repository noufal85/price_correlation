"""Clustering engine - multiple clustering methods for stock grouping."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
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


# =============================================================================
# Additional Clustering Methods for Large Datasets
# =============================================================================


def cluster_hdbscan(
    distance_matrix: np.ndarray,
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> np.ndarray:
    """
    HDBSCAN clustering with precomputed distances.

    Better than DBSCAN for varying density clusters.
    Automatically determines number of clusters.

    Args:
        distance_matrix: NxN distance matrix
        min_cluster_size: Minimum points to form a cluster
        min_samples: Core point neighborhood size

    Returns:
        Labels array (-1 for noise)
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(distance_matrix)


def cluster_hierarchical_forced_k(
    condensed_distance: np.ndarray,
    n_stocks: int,
    n_clusters: int | None = None,
    linkage_method: str = "ward",
) -> np.ndarray:
    """
    Hierarchical clustering with forced cluster count.

    Instead of optimizing k, uses a reasonable default based on dataset size.

    Args:
        condensed_distance: Condensed distance array from pdist
        n_stocks: Number of stocks being clustered
        n_clusters: Desired clusters (default: max(20, sqrt(n)))
        linkage_method: Linkage method (ward, average, complete)

    Returns:
        Labels array (0-indexed)
    """
    if n_clusters is None:
        # Default: sqrt(n) with minimum of 20
        n_clusters = max(20, int(np.sqrt(n_stocks)))

    Z = linkage(condensed_distance, method=linkage_method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels - 1  # Convert to 0-indexed


def cluster_kmeans_pca(
    returns_df: pd.DataFrame,
    n_clusters: int | None = None,
    n_components: int = 50,
) -> np.ndarray:
    """
    K-Means clustering on PCA-reduced return space.

    Reduces dimensionality first, then clusters.
    Much faster and often produces better separation.

    Args:
        returns_df: DataFrame of normalized returns (rows=dates, cols=tickers)
        n_clusters: Number of clusters (default: n_stocks / 50)
        n_components: PCA dimensions to reduce to

    Returns:
        Labels array (0-indexed)
    """
    n_stocks = returns_df.shape[1]

    # Auto-select k if not provided
    if n_clusters is None:
        n_clusters = max(20, n_stocks // 50)

    # Reduce dimensions (transpose: we want stocks as samples)
    n_comp = min(n_components, returns_df.shape[0] - 1, n_stocks - 1)
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(returns_df.values.T)

    # Cluster in reduced space
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        random_state=42,
    )
    return kmeans.fit_predict(reduced)


def cluster_louvain(
    corr_matrix: np.ndarray,
    threshold: float = 0.5,
    resolution: float = 1.0,
) -> np.ndarray:
    """
    Louvain community detection on correlation graph.

    Treats stocks as nodes, correlations as edge weights.
    Finds natural communities without specifying k.

    Args:
        corr_matrix: NxN correlation matrix
        threshold: Minimum correlation to create edge
        resolution: Higher = more clusters

    Returns:
        Labels array (0-indexed)
    """
    import networkx as nx
    from community import community_louvain

    n = corr_matrix.shape[0]

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges for correlations above threshold
    for i in range(n):
        for j in range(i + 1, n):
            corr = corr_matrix[i, j]
            if corr >= threshold:
                G.add_edge(i, j, weight=float(corr))

    # Handle disconnected nodes
    if G.number_of_edges() == 0:
        # No edges above threshold, return all as separate clusters
        return np.arange(n)

    # Detect communities
    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)

    # Convert to array
    labels = np.array([partition.get(i, -1) for i in range(n)])
    return labels


def run_multiple_clustering(
    distance_matrix: np.ndarray,
    condensed_distance: np.ndarray,
    corr_matrix: np.ndarray,
    returns_df: pd.DataFrame,
    methods: list[str],
    config: dict | None = None,
) -> dict:
    """
    Run multiple clustering methods and return all results.

    Args:
        distance_matrix: NxN distance matrix
        condensed_distance: Condensed distance array
        corr_matrix: NxN correlation matrix
        returns_df: Normalized returns DataFrame
        methods: List of methods to run
        config: Optional configuration overrides

    Returns:
        Dict mapping method name to results dict
    """
    config = config or {}
    n_stocks = distance_matrix.shape[0]
    results = {}

    for method in methods:
        print(f"  Running {method} clustering...", flush=True)

        try:
            if method == "hierarchical":
                # Forced-k hierarchical
                n_clusters = config.get("hierarchical_k") or max(20, int(np.sqrt(n_stocks)))
                linkage_method = config.get("hierarchical_linkage", "ward")
                labels = cluster_hierarchical_forced_k(
                    condensed_distance,
                    n_stocks,
                    n_clusters=n_clusters,
                    linkage_method=linkage_method,
                )

            elif method == "hdbscan":
                min_cluster_size = config.get("hdbscan_min_cluster_size", 10)
                min_samples = config.get("hdbscan_min_samples", 5)
                labels = cluster_hdbscan(
                    distance_matrix,
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                )

            elif method == "kmeans_pca":
                n_clusters = config.get("kmeans_n_clusters")
                n_components = config.get("kmeans_pca_components", 50)
                labels = cluster_kmeans_pca(
                    returns_df,
                    n_clusters=n_clusters,
                    n_components=n_components,
                )

            elif method == "louvain":
                threshold = config.get("louvain_threshold", 0.5)
                resolution = config.get("louvain_resolution", 1.0)
                labels = cluster_louvain(
                    corr_matrix,
                    threshold=threshold,
                    resolution=resolution,
                )

            elif method == "dbscan":
                eps = find_optimal_eps(distance_matrix)
                labels = cluster_dbscan(distance_matrix, eps=eps)

            else:
                print(f"  Unknown method: {method}, skipping", flush=True)
                continue

            # Compute stats
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))

            # Compute silhouette if we have valid clusters
            if n_clusters_found >= 2 and n_noise < len(labels):
                # For silhouette, exclude noise points
                valid_mask = labels >= 0
                if valid_mask.sum() >= 2:
                    try:
                        sil_score = silhouette_score(
                            distance_matrix[valid_mask][:, valid_mask],
                            labels[valid_mask],
                            metric="precomputed",
                        )
                    except Exception:
                        sil_score = 0.0
                else:
                    sil_score = 0.0
            else:
                sil_score = 0.0

            # Cluster size distribution
            unique_labels = [l for l in set(labels) if l >= 0]
            cluster_sizes = {int(l): int(np.sum(labels == l)) for l in unique_labels}

            results[method] = {
                "labels": labels,
                "n_clusters": n_clusters_found,
                "n_noise": n_noise,
                "silhouette": float(sil_score),
                "cluster_sizes": cluster_sizes,
            }

            print(f"    {method}: {n_clusters_found} clusters, {n_noise} noise, silhouette={sil_score:.4f}", flush=True)

        except Exception as e:
            print(f"    {method} failed: {e}", flush=True)
            results[method] = {
                "labels": np.zeros(n_stocks, dtype=int),
                "n_clusters": 1,
                "n_noise": 0,
                "silhouette": 0.0,
                "cluster_sizes": {0: n_stocks},
                "error": str(e),
            }

    return results
