import numpy as np
import faiss
import utils
import argparse
from scipy.sparse.csgraph import minimum_spanning_tree

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="K-means clustering for dataset partitioning.")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of clusters for K-means.")
    parser.add_argument("--db", type=str, required=True, help="Dataset name (e.g., 'sift').")
    args = parser.parse_args()

    num_clusters = args.num_clusters
    db = args.db

    if num_clusters <= 2:
        print('Error: num_clusters must be at least 3.')
        exit(1)

    # Load centroid data
    kmeans_centroids_path = f'/home/jlc/hnswlib/data/{db}/kmeans/multi-index-data/{num_clusters}parts/{db}_kmeans_centroids.fvecs'
    centroid_data = utils.fvecs_read(kmeans_centroids_path)  # Shape: (N, dim)
    N, dim = centroid_data.shape

    # Build FAISS index to hold centroids
    index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean)
    index.add(centroid_data)  # Add centroids to the index

    # Use centroids as both the query and the database
    kbase = N  # Number of neighbors to retrieve
    distances, assignments = index.search(centroid_data, kbase)

    # Convert distances to a full pairwise symmetric distance matrix
    pairwise_distances = np.zeros((N, N))
    for i in range(N):
        for j, dist in zip(assignments[i], distances[i]):
            pairwise_distances[i, j] = dist

    # Ensure the matrix is symmetric (since distances are symmetric)
    pairwise_distances = np.minimum(pairwise_distances, pairwise_distances.T)

    # Construct MST using scipy
    mst = minimum_spanning_tree(pairwise_distances)

    print("MST (Minimum Spanning Tree) adjacency matrix:")
    # Convert the sparse MST to a dense format for easier visualization
    mst_dense = mst.toarray()
    print(mst_dense)
    utils.fvecs_write(f'/home/jlc/hnswlib/data/{db}/kmeans/multi-index-data/{num_clusters}parts/{db}_centroid_MST.fvecs', mst_dense.astype(np.float32))