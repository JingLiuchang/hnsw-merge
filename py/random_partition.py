import numpy as np
import utils
import os
import argparse

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Random partitioning for dataset partitioning.")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of partitions for random partitioning.")
    parser.add_argument("--db", type=str, required=True, help="Dataset name (e.g., 'sift').")
    args = parser.parse_args()

    num_clusters = args.num_clusters
    db = args.db

    if num_clusters < 2:
        print('Error: num_clusters must be at least 2.')
        exit(1)
    elif num_clusters == 2:
        data_path = f'/mnt/ssd/merge_bench/{db}/{db}_base.fvecs'
        random_centroids_save_path = f'/mnt/ssd/merge_bench/{db}/random/bi-index-data/{db}_random_centroids.fvecs'
        partition_save_path = f'/mnt/ssd/merge_bench/{db}/random/bi-index-data/'
        base_save_path = f'/mnt/ssd/merge_bench/{db}/random/bi-index-data/{db}_random_base.fvecs'
    else:
        data_path = f'/mnt/ssd/merge_bench/{db}/{db}_base.fvecs'
        random_centroids_save_path = f'/mnt/ssd/merge_bench/{db}/random/multi-index-data/{db}_random_centroids.fvecs'
        partition_save_path = f'/mnt/ssd/merge_bench/{db}/random/multi-index-data/{num_clusters}parts/'
        partition_index_path = f'/mnt/ssd/merge_bench/{db}/random/multi-index-merged/{num_clusters}parts/'
        base_save_path = f'/mnt/ssd/merge_bench/{db}/random/multi-index-data/{num_clusters}parts/{db}_random_base.fvecs'
        os.makedirs(os.path.dirname(partition_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(partition_index_path), exist_ok=True)

    # Read data
    data = utils.fvecs_read(data_path)  # Shape: (N, dim)
    num_data, dim = data.shape

    # Shuffle data (shuffle rows)
    np.random.seed(42)  # For reproducibility
    perm = np.random.permutation(num_data)
    data = data[perm]

    # Save shuffled data to base_save_path
    utils.fvecs_write(base_save_path, data)

    # Calculate partition sizes
    partition_size = num_data // num_clusters
    remainder = num_data % num_clusters

    # Create random partitions by dividing shuffled data sequentially
    start_idx = 0
    for i in range(num_clusters):
        # Calculate end index for current partition
        current_partition_size = partition_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_partition_size

        # Get partition data
        partition_data = data[start_idx:end_idx]

        # Save partition data
        cluster_save_path = f'{partition_save_path}{db}_randomP{i+1}_base.fvecs'
        utils.fvecs_write(cluster_save_path, partition_data)

        start_idx = end_idx

    print("Random partitioning completed and data saved successfully.")