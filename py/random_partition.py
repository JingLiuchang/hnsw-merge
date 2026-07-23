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

    data_root = os.environ.get('DATA_PATH', '/mnt/ssd/merge_bench')
    dataset_path = os.path.join(data_root, db)

    if num_clusters < 2:
        print('Error: num_clusters must be at least 2.')
        exit(1)
    elif num_clusters == 2:
        data_path = os.path.join(dataset_path, f'{db}_base.fvecs')
        partition_save_path = os.path.join(dataset_path, 'random', 'bi-index-data')
        partition_index_path = os.path.join(dataset_path, 'random', 'bi-index-merged')
        base_save_path = os.path.join(partition_save_path, f'{db}_random_base.fvecs')
    else:
        data_path = os.path.join(dataset_path, f'{db}_base.fvecs')
        partition_save_path = os.path.join(
            dataset_path, 'random', 'multi-index-data', f'{num_clusters}parts')
        partition_index_path = os.path.join(
            dataset_path, 'random', 'multi-index-merged', f'{num_clusters}parts')
        base_save_path = os.path.join(partition_save_path, f'{db}_random_base.fvecs')

    os.makedirs(partition_save_path, exist_ok=True)
    os.makedirs(partition_index_path, exist_ok=True)

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
        cluster_save_path = os.path.join(
            partition_save_path, f'{db}_randomP{i+1}_base.fvecs')
        utils.fvecs_write(cluster_save_path, partition_data)

        start_idx = end_idx

    original_gt_path = os.path.join(dataset_path, f'{db}_groundtruth.ivecs')
    if os.path.exists(original_gt_path):
        original_gt = utils.ivecs_read(original_gt_path)
        inverse_perm = np.empty(num_data, dtype=np.int64)
        inverse_perm[perm] = np.arange(num_data, dtype=np.int64)
        shuffled_gt = inverse_perm[original_gt].astype(np.int32, copy=False)
        gt_save_path = os.path.join(
            partition_save_path, f'{db}_random_groundtruth.ivecs')
        utils.ivecs_write(gt_save_path, shuffled_gt)
        print(f"Remapped ground truth saved to {gt_save_path}.")

    print("Random partitioning completed and data saved successfully.")
