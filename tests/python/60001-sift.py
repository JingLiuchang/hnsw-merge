import hnswlib
import numpy as np
import pickle
import format_trans
from time import time
import os

def calculate_recall(gt, predicted_labels, k=10):
    """
    计算 Recall@K

    参数:
    gt: 形状为 (N, K) 的 numpy 数组，表示 N 个数据的真实最近邻 ID
    predicted_labels: 形状为 (N, k) 的 numpy 数组，表示查询得到的最近邻 ID
    k: 考虑前 k 个结果

    返回:
    float: 平均 Recall@K 值
    """
    n = gt.shape[0]
    recall = 0.0

    for i in range(n):
        # 获取真实的前 k 个最近邻
        ground_truth = set(gt[i, :k])
        # 获取预测的前 k 个最近邻
        predictions = set(predicted_labels[i, :k])
        # 计算交集大小
        intersection = ground_truth.intersection(predictions)
        # 计算单个查询的 recall
        recall += len(intersection) / k

    # 计算平均 recall
    return recall / n

dim = 128
num_elements = 1000000

# Generating sample data
data = format_trans.fvecs_read('/home/jlc/datasets/sift/sift_base.fvecs')
gt = format_trans.ivecs_read('/home/jlc/datasets/sift/sift_groundtruth.ivecs')
query = format_trans.fvecs_read('/home/jlc/datasets/sift/sift_query.fvecs')
assert(data.shape[0] == num_elements)
ids = np.arange(num_elements)
index_path='sift-L64-R32.bin'

p = hnswlib.Index(space = 'l2', dim = dim)

if os.path.exists(index_path):
    print("Loading index from '%s'" % index_path)
    p.load_index(index_path, max_elements = num_elements)
else:
    print("Creating new index")
    p.init_index(max_elements = num_elements, ef_construction = 64, M = 32)

    s = time()
    p.add_items(data, ids, num_threads=1)
    e = time()
    print(f'indexing time: {e-s:.2f}s')


    print("Saving index to '%s'" % index_path)
    p.save_index(index_path)


#indexing time: 125.52s

print("e,queries_per_second,recall")
for e in range(10, 310, 10):  # Note: changed range from (10,10,300) to (10,310,10)
    p.set_ef(e)
    s = time()
    labels, distances = p.knn_query(query, k=10, num_threads=1)
    e_time = time()  # Renamed variable to avoid conflict with loop variable 'e'
    recall = calculate_recall(gt, labels, k=10)
    QPS = query.shape[0] / (e_time - s)

    # Format output as requested
    print(f"{e},{QPS:.4f},{recall:.4f}")