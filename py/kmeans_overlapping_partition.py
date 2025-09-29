import numpy as np
import faiss
import utils
import os
import argparse
import struct

# Parameters
sample_size = 256000

def save_assignments_to_bin(assignments, filename):
    """Save assignments list to binary file"""
    with open(filename, 'wb') as f:
        # Write number of assignments (unsigned int)
        f.write(struct.pack('I', len(assignments)))
        # Write each assignment (size_t id, unsigned int cid)
        for id_val, cid in assignments:
            f.write(struct.pack('QI', id_val, cid))  # Q=size_t, I=unsigned int

def save_idmaps_to_bin(idmaps, filename):
    """Save idmaps list[list] to binary file"""
    with open(filename, 'wb') as f:
        # Write number of centroids (unsigned int)
        f.write(struct.pack('I', len(idmaps)))
        # Write each centroid's idmap
        for idmap in idmaps:
            # Write size of this idmap (size_t)
            f.write(struct.pack('Q', len(idmap)))
            # Write each mapping (size_t global_id)
            for global_id in idmap:
                f.write(struct.pack('Q', global_id))

def save_globalid_map_to_bin(filename, global_to_local_map):
    # 保存为二进制文件
    with open(filename, "wb") as f:
        # 写入总大小（size_t）
        f.write(struct.pack("Q", len(global_to_local_map)))  # Q: size_t (unsigned long long)

        # 写入每个 global_id 的数据
        for local_map in global_to_local_map:
            # 写入字典大小（unsigned）
            f.write(struct.pack("I", len(local_map)))  # I: unsigned int

            # 写入字典中的每个键值对 (cid, local_id)
            for cid, local_id in local_map.items():
                f.write(struct.pack("I", cid))       # I: unsigned int (4 bytes)
                f.write(struct.pack("Q", local_id))  # Q: size_t (8 bytes)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="K-means clustering for dataset partitioning.")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of clusters for K-means.")
    parser.add_argument("--db", type=str, required=True, help="Dataset name (e.g., 'sift').")
    parser.add_argument("--kbase", type=int, default=2, help="Assign each data to kbase centroids.")
    args = parser.parse_args()

    num_clusters = args.num_clusters
    db = args.db
    kbase = args.kbase

    if num_clusters <= 2:
        print('Error: num_clusters must be at least 3.')
        exit(1)
    else:
        data_path = f'/home/jlc/hnswlib/data/{db}/{db}_base.fvecs'
        kmeans_centroids_save_path = f'/home/jlc/hnswlib/data/{db}/overlap/multi-index-data/{num_clusters}parts/{db}_kmeans_centroids.fvecs'
        partition_save_path = f'/home/jlc/hnswlib/data/{db}/overlap/multi-index-data/{num_clusters}parts/'
        partition_index_path = f'/home/jlc/hnswlib/data/{db}/overlap/multi-index-merged/{num_clusters}parts/'
        performance_path = f'/home/jlc/hnswlib/data/{db}/overlap/performance/{num_clusters}parts/'
        os.makedirs(os.path.dirname(partition_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(partition_index_path), exist_ok=True)

    # Read data
    data = utils.fvecs_read(data_path)  # Shape: (N, dim)
    num_data, dim = data.shape

    # Shuffle data (shuffle rows)
    np.random.seed(42)  # For reproducibility
    perm = np.random.permutation(num_data)
    data = data[perm]
    utils.fvecs_write(f'/home/jlc/hnswlib/data/{db}/overlap/multi-index-data/{num_clusters}parts/{db}_shuffle_base.fvecs', data)

    # Use a subset of data for training (min(sample_size, num_data))
    train_data = data[:min(sample_size, num_data)]

    # Perform k-means clustering with faiss
    kmeans = faiss.Kmeans(d=dim, k=num_clusters, niter=20, verbose=True, gpu=True, max_points_per_centroid=int(sample_size / num_clusters))
    kmeans.train(train_data)

    # Save centroids using utils.fvecs_write
    utils.fvecs_write(kmeans_centroids_save_path, kmeans.centroids)

    # Assign each data point to kbase clusters (Merge order selection)
    print(f"Assigning each data point to {kbase} nearest clusters...")

    # Search for kbase nearest centroids for each data point
    distances, assignments = kmeans.index.search(data.astype(np.float32), kbase)

    # Build assignments list: [(id, cid), (id, cid), ...]
    assignments_list = []
    for data_id in range(num_data):
        for k in range(kbase):
            cid = assignments[data_id, k]
            assignments_list.append((data_id, cid))

    # Save assignments to binary file
    assignments_file = os.path.join(partition_save_path, f'{db}_kbase{kbase}_assignments.bin')
    save_assignments_to_bin(assignments_list, assignments_file)
    print(f"Assignments saved to {assignments_file}")

    # Build k datasets for each centroid and id mappings
    print("Building datasets for each centroid...")

    # Initialize data structures
    centroid_data = [[] for _ in range(num_clusters)]  # Store data vectors for each centroid
    idmaps = [[] for _ in range(num_clusters)]         # Store global_id mapping for each centroid

    # Collect data for each centroid
    for data_id in range(num_data):
        for k in range(kbase):
            cid = assignments[data_id, k]
            centroid_data[cid].append(data[data_id])
            idmaps[cid].append(data_id)

    # Save datasets and build final idmaps
    for cid in range(num_clusters):
        if len(centroid_data[cid]) > 0:
            # Convert to numpy array and save as fvecs
            centroid_vectors = np.array(centroid_data[cid], dtype=np.float32)
            centroid_file = os.path.join(partition_save_path, f'{db}_centroid_{cid+1}.fvecs')
            utils.fvecs_write(centroid_file, centroid_vectors)
            print(f"Centroid {cid}: {centroid_vectors.shape} vectors saved to {centroid_file}")
        else:
            print(f"Warning: Centroid {cid} has no assigned data points")

    # Save idmaps to binary file
    idmaps_file = os.path.join(partition_save_path, f'{db}_idmaps_kbase{kbase}.bin')
    save_idmaps_to_bin(idmaps, idmaps_file)
    print(f"ID mappings saved to {idmaps_file}")

    # build map from global_id to (cid, local_id)
    global_to_local_map = [{} for _ in range(num_data)]  # list of dicts
    for cid in range(num_clusters):
        for local_id, global_id in enumerate(idmaps[cid]):
            global_to_local_map[global_id][cid] = local_id
    global2local_file = os.path.join(partition_save_path, f'{db}_global2local_kbase{kbase}.bin')
    save_globalid_map_to_bin(global2local_file, global_to_local_map)
    print(f"Global to local ID map saved to {global2local_file}")

    # Print summary
    print(f"\nSummary:")
    print(f"Total data points: {num_data}")
    print(f"Number of clusters: {num_clusters}")
    print(f"Each point assigned to {kbase} clusters")
    print(f"Total assignments: {len(assignments_list)}")

    for cid in range(num_clusters):
        print(f"Centroid {cid}: {len(idmaps[cid])} data points")