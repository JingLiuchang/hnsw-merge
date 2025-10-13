import numpy as np
import faiss
import utils
import os
import argparse
import struct
from time import time

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
    analysis = False  # Set to True to enable analysis

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

    if not analysis:
        print('#################################################################################')
        exit(0)

    # 重叠情况
    print("\n" + "="*80)
    print("Overlap Analysis for Each Cluster")
    print("="*80)

    for cid in range(num_clusters):
        if len(idmaps[cid]) == 0:
            continue

        # 统计该聚类中每个数据点的出现次数
        exclusive_count = 0  # 独有的数据点
        shared_count = 0     # 共享的数据点
        shared_with = {}     # 与哪些聚类共享: {other_cid: count}

        for global_id in idmaps[cid]:
            # 查看这个global_id出现在多少个聚类中
            clusters_containing = global_to_local_map[global_id]

            if len(clusters_containing) == 1:
                # 只在当前聚类中
                exclusive_count += 1
            else:
                # 被多个聚类共享
                shared_count += 1
                # 统计与哪些聚类共享
                for other_cid in clusters_containing.keys():
                    if other_cid != cid:
                        shared_with[other_cid] = shared_with.get(other_cid, 0) + 1

        total = len(idmaps[cid])
        print(f"\nCluster {cid+1}:")
        print(f"  Total vectors: {total}")
        print(f"  Exclusive (only in this cluster): {exclusive_count} ({exclusive_count/total*100:.2f}%)")
        print(f"  Shared (in multiple clusters): {shared_count} ({shared_count/total*100:.2f}%)")

        if shared_with:
            print(f"  Shared with other clusters:")
            # 按共享数量排序
            sorted_shared = sorted(shared_with.items(), key=lambda x: x[1], reverse=True)
            for other_cid, count in sorted_shared:  # 只显示前5个
                print(f"    - Cluster {other_cid+1}: {count} vectors ({count/total*100:.2f}%)")

    # 在代码末尾添加以下分析代码

    print("\n" + "="*80)
    print("Top-K Centroid Assignment Analysis")
    print("="*80)

    # 构建每个数据点的排序聚类列表
    # data_point_rankings[global_id] = [(cid, rank), (cid, rank), ...]
    data_point_rankings = [[] for _ in range(num_data)]

    for data_id in range(num_data):
        for rank in range(kbase):
            cid = assignments[data_id, rank]
            data_point_rankings[data_id].append((cid, rank))

    # 为每个聚类分析：当它是Top-1时，Top-2的分布
    print("\nFor each cluster as Top-1, analyzing Top-2 distribution:")
    print("="*80)

    # 存储分析结果
    top1_to_top2_distribution = {}

    for cid in range(num_clusters):
        # 找到所有将cid作为Top-1的数据点
        top1_points = []
        for data_id in range(num_data):
            if len(data_point_rankings[data_id]) > 0:
                top1_cid, rank = data_point_rankings[data_id][0]
                if top1_cid == cid and rank == 0:
                    top1_points.append(data_id)

        if len(top1_points) == 0:
            continue

        # 统计这些点的Top-2选择
        top2_distribution = {}
        for data_id in top1_points:
            if len(data_point_rankings[data_id]) > 1:
                top2_cid, rank = data_point_rankings[data_id][1]
                if rank == 1:  # 确保是Top-2
                    top2_distribution[top2_cid] = top2_distribution.get(top2_cid, 0) + 1

        top1_to_top2_distribution[cid] = {
            'total': len(top1_points),
            'top2_dist': top2_distribution
        }

        print(f"\nCluster {cid+1} (as Top-1 choice):")
        print(f"  Number of points choosing this as closest: {len(top1_points)}")

        if top2_distribution:
            print(f"  Their Top-2 choices:")
            # 按数量排序
            sorted_top2 = sorted(top2_distribution.items(), key=lambda x: x[1], reverse=True)
            for top2_cid, count in sorted_top2[:10]:  # 显示前10个
                percentage = count / len(top1_points) * 100
                print(f"    Cluster {top2_cid+1}: {count} points ({percentage:.2f}%)")

            if len(sorted_top2) > 10:
                print(f"    ... and {len(sorted_top2) - 10} more clusters")
        else:
            print(f"  No Top-2 data (kbase might be 1)")

    # 更通用的版本：分析Top-k到Top-(k+1)的转移
    print("\n" + "="*80)
    print("General Top-k to Top-(k+1) Transition Analysis")
    print("="*80)

    for current_rank in range(min(kbase - 1, 3)):  # 分析前3个rank的转移
        print(f"\n--- Rank {current_rank+1} to Rank {current_rank+2} Transitions ---\n")

        for cid in range(num_clusters):
            # 找到所有在rank=current_rank时选择cid的数据点
            rank_k_points = []
            for data_id in range(num_data):
                if len(data_point_rankings[data_id]) > current_rank:
                    if data_point_rankings[data_id][current_rank][0] == cid:
                        rank_k_points.append(data_id)

            if len(rank_k_points) == 0:
                continue

            # 统计这些点在下一个rank的选择
            next_rank_distribution = {}
            for data_id in rank_k_points:
                if len(data_point_rankings[data_id]) > current_rank + 1:
                    next_cid = data_point_rankings[data_id][current_rank + 1][0]
                    next_rank_distribution[next_cid] = next_rank_distribution.get(next_cid, 0) + 1

            print(f"Cluster {cid+1} (at rank {current_rank+1}): {len(rank_k_points)} points")
            # if next_rank_distribution and len(next_rank_distribution) <= 5:
            #     sorted_next = sorted(next_rank_distribution.items(), key=lambda x: x[1], reverse=True)
            #     for next_cid, count in sorted_next:
            #         print(f"  → Cluster {next_cid+1}: {count} ({count/len(rank_k_points)*100:.1f}%)")
            # elif next_rank_distribution:
            #     top3 = sorted(next_rank_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
            #     for next_cid, count in top3:
            #         print(f"  → Cluster {next_cid+1}: {count} ({count/len(rank_k_points)*100:.1f}%)")
            sorted_next = sorted(next_rank_distribution.items(), key=lambda x: x[1], reverse=True)
            for next_cid, count in sorted_next:
                print(f"  → Cluster {next_cid+1}: {count} ({count/len(rank_k_points)*100:.1f}%)")

    # 保存详细的转移矩阵
    print("\n" + "="*80)
    print("Generating Transition Matrices")
    print("="*80)

    # Top-1 到 Top-2 的转移矩阵
    transition_matrix_1to2 = np.zeros((num_clusters, num_clusters), dtype=int)

    for data_id in range(num_data):
        if len(data_point_rankings[data_id]) >= 2:
            top1_cid = data_point_rankings[data_id][0][0]
            top2_cid = data_point_rankings[data_id][1][0]
            transition_matrix_1to2[top1_cid][top2_cid] += 1

    # 保存转移矩阵
    transition_file = os.path.join(partition_save_path, f'{db}_transition_1to2_kbase{kbase}.npy')
    np.save(transition_file, transition_matrix_1to2)
    print(f"Top-1 to Top-2 transition matrix saved to {transition_file}")

    # 打印转移矩阵（部分）
    print("\nTop-1 to Top-2 Transition Matrix (first 10x10):")
    print("Rows: Top-1 cluster, Columns: Top-2 cluster")
    print()
    print("       ", end="")
    for j in range(min(10, num_clusters)):
        print(f"C{j+1:4d} ", end="")
    print()

    for i in range(min(10, num_clusters)):
        print(f"C{i+1:4d}: ", end="")
        for j in range(min(10, num_clusters)):
            print(f"{transition_matrix_1to2[i][j]:5d} ", end="")
        print()

    # 保存详细的文本报告
    report_file = os.path.join(partition_save_path, f'{db}_topk_transition_report_kbase{kbase}.txt')
    with open(report_file, 'w') as f:
        f.write("Top-K Centroid Assignment Transition Report\n")
        f.write("="*80 + "\n\n")

        for cid in range(num_clusters):
            if cid in top1_to_top2_distribution:
                info = top1_to_top2_distribution[cid]
                f.write(f"Cluster {cid+1} (as Top-1 choice):\n")
                f.write(f"  Total points: {info['total']}\n")
                f.write(f"  Top-2 distribution:\n")

                sorted_dist = sorted(info['top2_dist'].items(), key=lambda x: x[1], reverse=True)
                for top2_cid, count in sorted_dist:
                    pct = count / info['total'] * 100
                    f.write(f"    Cluster {top2_cid+1}: {count:6d} ({pct:5.2f}%)\n")
                f.write("\n")

    print(f"\nDetailed transition report saved to {report_file}")

    # 可视化：找出最强的Top-1 -> Top-2 转移对
    print("\n" + "="*80)
    print("Strongest Top-1 → Top-2 Transitions")
    print("="*80)

    transitions = []
    for i in range(num_clusters):
        for j in range(num_clusters):
            if transition_matrix_1to2[i][j] > 0:
                transitions.append((i, j, transition_matrix_1to2[i][j]))

    transitions.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 20 strongest transitions:")
    for i, (cid1, cid2, count) in enumerate(transitions[:20], 1):
        print(f"{i:2d}. Cluster {cid1+1} → Cluster {cid2+1}: {count:6d} points")

    print("\n" + "="*80)
    print("Building Minimum Graph with Diameter <= 2")
    print("="*80)

    # 调用函数 - 接收新增的返回值
    edge_count, inserted_edges, final_adj_matrix, final_dist_matrix, \
        last_edge_weight, min_edge_weight, max_edge_weight = \
        utils.build_graph_until_diameter_2_fast(transition_matrix_1to2)

    # # 保存结果到文件 - 包含边权重信息
    # result_file = os.path.join(partition_save_path, f'{db}_diameter2_graph_kbase{kbase}.npz')
    # np.savez(result_file,
    #          edge_count=edge_count,
    #          adj_matrix=final_adj_matrix,
    #          dist_matrix=final_dist_matrix,
    #          edges=np.array(inserted_edges, dtype=object),
    #          last_edge_weight=last_edge_weight,
    #          min_edge_weight=min_edge_weight,
    #          max_edge_weight=max_edge_weight)
    # print(f"\nGraph data saved to {result_file}")

    # 输出插入的边的详细信息
    print("\n" + "="*80)
    print("Inserted Edges Details")
    print("="*80)

    # 显示权重范围
    print(f"\nEdge weight range:")
    print(f"  Maximum weight: {max_edge_weight:,}")
    print(f"  Minimum weight: {min_edge_weight:,}")
    print(f"  Last inserted: {last_edge_weight:,}")
    print(f"  Weight ratio (max/min): {max_edge_weight/min_edge_weight:.2f}x" if min_edge_weight > 0 else "  Weight ratio: N/A")

    # 边权重分布
    edge_weights = [w for _, _, w in inserted_edges]
    edge_weights_sorted = sorted(edge_weights, reverse=True)

    print(f"\nEdge weight distribution:")
    percentiles = [0, 25, 50, 75, 100]
    for p in percentiles:
        idx = int(len(edge_weights_sorted) * p / 100)
        if idx >= len(edge_weights_sorted):
            idx = len(edge_weights_sorted) - 1
        print(f"  {p:3d}th percentile: {edge_weights_sorted[idx]:,}")

    print(f"\nTop 30 inserted edges (by weight):")
    for idx, (i, j, weight) in enumerate(inserted_edges[:30], 1):
        print(f"{idx:3d}. Cluster {i+1:3d} ↔ Cluster {j+1:3d}: weight={weight:8,}")

    if len(inserted_edges) > 30:
        print(f"\n... (showing 30 of {len(inserted_edges)} edges)")

        # 显示最后几条边（权重最小的）
        print(f"\nLast 10 inserted edges (smallest weights):")
        for idx, (i, j, weight) in enumerate(inserted_edges[-10:], len(inserted_edges)-9):
            print(f"{idx:3d}. Cluster {i+1:3d} ↔ Cluster {j+1:3d}: weight={weight:8,}")
