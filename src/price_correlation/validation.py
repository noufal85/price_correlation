"""Validation - evaluate clustering quality and visualize results."""

from collections import Counter

import numpy as np
from sklearn.metrics import silhouette_score


def compute_silhouette(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute silhouette score for clustering.

    Returns:
        Score in range [-1, 1], higher is better
    """
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) < 2:
        return 0.0

    mask = labels != -1
    if mask.sum() < 2:
        return 0.0

    filtered_dist = distance_matrix[mask][:, mask]
    filtered_labels = labels[mask]

    return silhouette_score(filtered_dist, filtered_labels, metric="precomputed")


def compute_cluster_stats(
    labels: np.ndarray,
    tickers: list[str],
) -> dict:
    """
    Compute statistics about clustering results.

    Returns:
        Dictionary with cluster statistics
    """
    label_counts = Counter(labels)

    noise_count = label_counts.pop(-1, 0)
    cluster_ids = sorted(label_counts.keys())

    sizes = [label_counts[i] for i in cluster_ids]

    return {
        "n_clusters": len(cluster_ids),
        "n_noise": noise_count,
        "n_total": len(labels),
        "cluster_sizes": dict(label_counts),
        "largest_cluster": max(sizes) if sizes else 0,
        "smallest_cluster": min(sizes) if sizes else 0,
        "mean_cluster_size": np.mean(sizes) if sizes else 0,
    }


def get_cluster_members(
    labels: np.ndarray,
    tickers: list[str],
) -> dict[int, list[str]]:
    """
    Get ticker members for each cluster.

    Returns:
        {cluster_id: [ticker1, ticker2, ...]}
    """
    clusters: dict[int, list[str]] = {}

    for ticker, label in zip(tickers, labels):
        label_int = int(label)
        if label_int not in clusters:
            clusters[label_int] = []
        clusters[label_int].append(ticker)

    for cluster_id in clusters:
        clusters[cluster_id].sort()

    return dict(sorted(clusters.items()))


def generate_tsne_plot(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    tickers: list[str],
    output_path: str,
) -> None:
    """
    Generate 2D t-SNE visualization of clusters.

    Saves plot to output_path.
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=min(30, len(labels) - 1),
        random_state=42,
    )
    embedding = tsne.fit_transform(distance_matrix)

    plt.figure(figsize=(12, 8))

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        color = "gray" if label == -1 else colors[idx]
        marker = "x" if label == -1 else "o"
        alpha = 0.3 if label == -1 else 0.7
        label_name = "Noise" if label == -1 else f"Cluster {label}"

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            marker=marker,
            alpha=alpha,
            label=label_name,
            s=50,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Stock Clusters Visualization")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_cluster_summary(
    labels: np.ndarray,
    tickers: list[str],
    max_display: int = 10,
) -> None:
    """Print a summary of clusters."""
    members = get_cluster_members(labels, tickers)
    stats = compute_cluster_stats(labels, tickers)

    print(f"\nClustering Summary")
    print(f"=" * 50)
    print(f"Total stocks: {stats['n_total']}")
    print(f"Clusters: {stats['n_clusters']}")
    print(f"Noise: {stats['n_noise']}")
    print(f"Mean cluster size: {stats['mean_cluster_size']:.1f}")
    print()

    for cluster_id, cluster_tickers in members.items():
        if cluster_id == -1:
            label = "NOISE"
        else:
            label = f"Cluster {cluster_id}"

        display = cluster_tickers[:max_display]
        extra = len(cluster_tickers) - max_display

        print(f"{label} ({len(cluster_tickers)} stocks):")
        print(f"  {', '.join(display)}", end="")
        if extra > 0:
            print(f" ... +{extra} more")
        else:
            print()
