import numpy as np
import faiss
import utils
import os
import argparse

# Parameters
sample_size = 256000

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="K-means clustering for dataset partitioning.")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of clusters for K-means.")
    parser.add_argument("--db", type=str, required=True, help="Dataset name (e.g., 'sift').")
    args = parser.parse_args()

    num_clusters = args.num_clusters
    db = args.db

    if num_clusters < 2:
        print('Error: num_clusters must be at least 2.')
        exit(1)
    elif num_clusters == 2:
        data_path = f'/mnt/ssd/merge_bench/{db}/{db}_base.fvecs'
        kmeans_centroids_save_path = f'/mnt/ssd/merge_bench/{db}/kmeans/bi-index-data/{db}_kmeans_centroids.fvecs'
        partition_save_path = f'/mnt/ssd/merge_bench/{db}/kmeans/bi-index-data/'
        base_save_path = f'/mnt/ssd/merge_bench/{db}/kmeans/bi-index-data/{db}_kmeans_base.fvecs'
    else:
        data_path = f'/mnt/ssd/merge_bench/{db}/{db}_base.fvecs'
        kmeans_centroids_save_path = f'/mnt/ssd/merge_bench/{db}/kmeans/multi-index-data/{num_clusters}parts/{db}_kmeans_centroids.fvecs'
        partition_save_path = f'/mnt/ssd/merge_bench/{db}/kmeans/multi-index-data/{num_clusters}parts/'
        partition_index_path = f'/mnt/ssd/merge_bench/{db}/kmeans/multi-index-merged/{num_clusters}parts/'
        base_save_path = f'/mnt/ssd/merge_bench/{db}/kmeans/multi-index-data/{num_clusters}parts/{db}_kmeans_base.fvecs'
        os.makedirs(os.path.dirname(partition_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(partition_index_path), exist_ok=True)

    # Read data
    data = utils.fvecs_read(data_path)  # Shape: (N, dim)
    num_data, dim = data.shape

    # Shuffle data (shuffle rows)
    np.random.seed(42)  # For reproducibility
    perm = np.random.permutation(num_data)
    data = data[perm]

    # Use a subset of data for training (min(sample_size, num_data))
    train_data = data[:min(sample_size, num_data)]

    # Perform k-means clustering with faiss
    kmeans = faiss.Kmeans(d=dim, k=num_clusters, niter=20, verbose=True, gpu=True, max_points_per_centroid=int(sample_size / num_clusters))
    kmeans.train(train_data)

    # Save centroids using utils.fvecs_write
    utils.fvecs_write(kmeans_centroids_save_path, kmeans.centroids)

    # Assign each data point to a cluster
    distances, assignments = kmeans.index.search(data, 1)  # Shape: (N, 1)
    assignments = assignments.flatten()

    # Save partitions and merge them into a single file
    sorted_data = []

    for i in range(num_clusters):
        # Get all data points belonging to cluster i
        cluster_data = data[assignments == i]

        # Save cluster data to f'/mnt/ssd/merge_bench/{db}/kmeans/bi-index-data/{db}_kmeansP{i+1}_base.fvecs'
        cluster_save_path = f'{partition_save_path}{db}_kmeansP{i+1}_base.fvecs'
        utils.fvecs_write(cluster_save_path, cluster_data)

        # Collect data for merging
        sorted_data.append(cluster_data)

    # Merge all partitions and save to base_save_path
    sorted_data = np.vstack(sorted_data)
    utils.fvecs_write(base_save_path, sorted_data)

    print("K-means clustering completed and data saved successfully.")