import numpy as np

import struct
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_u8bin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.uint8,
                          offset=start_idx * 1 * dim)
    return arr.reshape(nvecs, dim)


def write_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)


def write_fbin_nocp(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<I', nvecs))
        f.write(struct.pack('<I', dim))
        vecs.astype('float32',copy=False).tofile(f)

# 示例用法
# vecs = np.random.rand(10, 3).astype('float32')
# write_fbin_nocp('output.fbin', vecs)

def write_ibin(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('int32').flatten().tofile(f)

def write_ibin_nocp(filename, vecs):
    """ Write an array of int32 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<I', nvecs))
        f.write(struct.pack('<I', dim))
        vecs.astype('int32',copy=False).tofile(f)


def write_u8bin(filename: str, vecs: np.ndarray, ) -> None:
    """
    将uint8矩阵保存为SIFT1B BIGANN的bvecs格式。

    Args:
        matrix (np.ndarray): 要保存的uint8矩阵。
            可以是numpy数组或嵌套列表。
        filename (str): 输出文件的名称。

    Raises:
        ValueError: 如果输入矩阵不是2维或者数据类型不是uint8。
    """

    # 检查矩阵是否为2维uint8类型
    if vecs.ndim != 2 or vecs.dtype != np.uint8:
        raise ValueError("输入矩阵必须是2维uint8类型")

    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape

        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('uint8').flatten().tofile(f)

def write_u8bin_nocp(filename: str, vecs: np.ndarray, ) -> None:
    """
    将uint8矩阵保存为SIFT1B BIGANN的bvecs格式。

    Args:
        matrix (np.ndarray): 要保存的uint8矩阵。
            可以是numpy数组或嵌套列表。
        filename (str): 输出文件的名称。

    Raises:
        ValueError: 如果输入矩阵不是2维或者数据类型不是uint8。
    """

    # 检查矩阵是否为2维uint8类型
    if vecs.ndim != 2 or vecs.dtype != np.uint8:
        raise ValueError("输入矩阵必须是2维uint8类型")

    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape

        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('uint8', copy=False).tofile(f)

def bvecs_read(fname):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy()


def bvecs_write(fname, m):
    assert m.dtype == np.uint8, "Input matrix must be of dtype uint8"
    with open(fname, "wb") as f:
        dimension = [len(m[0])]

        for x in m:
            f.write(struct.pack('i' * len(dimension), *dimension))
            f.write(struct.pack('B' * len(x), *x))
import numpy as np
from collections import deque
import heapq

def bfs_shortest_paths(adj_matrix, num_nodes):
    """
    使用BFS计算所有点对之间的最短路径
    返回: 最短路径矩阵 (INF表示不可达)
    """
    INF = float('inf')
    dist = np.full((num_nodes, num_nodes), INF, dtype=float)

    # 初始化对角线为0
    for i in range(num_nodes):
        dist[i][i] = 0

    # 从每个节点开始BFS
    for start in range(num_nodes):
        queue = deque([start])
        visited = {start}
        dist[start][start] = 0

        while queue:
            u = queue.popleft()
            for v in range(num_nodes):
                if adj_matrix[u][v] > 0 and v not in visited:
                    visited.add(v)
                    dist[start][v] = dist[start][u] + 1
                    queue.append(v)

    return dist

def is_connected(adj_matrix, num_nodes):
    """快速检查图是否连通（使用DFS）"""
    if num_nodes == 0:
        return True

    visited = set()
    stack = [0]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)

        for neighbor in range(num_nodes):
            if adj_matrix[node][neighbor] > 0 and neighbor not in visited:
                stack.append(neighbor)

    return len(visited) == num_nodes

def check_diameter_at_most_2(dist_matrix, num_nodes):
    """
    检查图的直径是否<=2
    即所有连通的点对之间的最短路径是否<=2
    """
    INF = float('inf')
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if dist_matrix[i][j] != INF and dist_matrix[i][j] > 2:
                return False
    return True

"""
优化版本：使用增量更新减少重复计算
"""
def build_graph_until_diameter_2_fast(transition_matrix):
    """
    优化版本：使用增量更新减少重复计算
    按边权重从大到小插入边，直到图的直径<=2

    参数:
        transition_matrix: numpy数组，形状为(num_clusters, num_clusters)
                          transition_matrix[i][j] 表示从聚类i到聚类j的转移权重

    返回:
        edge_count: 插入的边数
        inserted_edges: 插入的边列表 [(i, j, weight), ...]
        adj_matrix: 最终的邻接矩阵
        dist_matrix: 最终的最短路径矩阵
        last_edge_weight: 最后插入的边的权重
        min_edge_weight: 所有插入边的最小权重
        max_edge_weight: 所有插入边的最大权重
    """
    num_clusters = transition_matrix.shape[0]
    adj_matrix = np.zeros((num_clusters, num_clusters), dtype=int)

    # 收集所有边
    edges = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            # 无向边：取两个方向的最大值
            weight = max(transition_matrix[i][j], transition_matrix[j][i])
            if weight > 0:
                edges.append((i, j, weight))

    # 按权重降序排序
    edges.sort(key=lambda x: x[2], reverse=True)

    print(f"\nTotal potential edges: {len(edges)}")
    print(f"Building graph until diameter <= 2...\n")

    inserted_edges = []
    edge_count = 0
    check_interval = 10
    last_edge_weight = 0

    for i, j, weight in edges:
        # 插入边（无向）
        adj_matrix[i][j] = weight
        adj_matrix[j][i] = weight
        edge_count += 1
        inserted_edges.append((i, j, weight))
        last_edge_weight = weight  # 更新最后插入边的权重

        # 动态调整检查间隔
        if edge_count <= 50:
            should_check = True
        elif edge_count <= 200:
            should_check = (edge_count % 5 == 0)
        else:
            should_check = (edge_count % check_interval == 0)

        if should_check:
            if is_connected(adj_matrix, num_clusters):
                dist_matrix = bfs_shortest_paths(adj_matrix, num_clusters)

                # 快速检查最大距离
                max_dist = 0
                INF = float('inf')
                for i in range(num_clusters):
                    for j in range(i + 1, num_clusters):
                        if dist_matrix[i][j] != INF:
                            max_dist = max(max_dist, dist_matrix[i][j])

                if max_dist <= 3:
                    # 计算边权重统计
                    min_edge_weight = min(e[2] for e in inserted_edges)
                    max_edge_weight = max(e[2] for e in inserted_edges)
                    sum_edge_weight = sum(e[2] for e in inserted_edges)

                    print(f"\n✓ Graph diameter = {max_dist} <= 2!")
                    print(f"  Total edges inserted: {edge_count}")
                    print(f"  Percentage of total edges: {edge_count/len(edges)*100:.2f}%")
                    print(f"\n  Edge weight statistics:")
                    print(f"    Last inserted edge weight: {last_edge_weight}")
                    print(f"    Maximum edge weight: {max_edge_weight}")
                    print(f"    Minimum edge weight: {min_edge_weight}")
                    print(f"    Average edge weight: {sum(e[2] for e in inserted_edges)/len(inserted_edges):.2f}")
                    print(f"    Sum of edge weights: {sum_edge_weight}")

                    # 输出最短路径统计
                    path_lengths = {0: 0, 1: 0, 2: 0, 3: 0}
                    for i in range(num_clusters):
                        for j in range(i + 1, num_clusters):
                            d = dist_matrix[i][j]
                            if d != INF:
                                path_lengths[int(d)] += 1

                    total_pairs = num_clusters * (num_clusters - 1) // 2
                    print(f"\n  Path length distribution:")
                    print(f"    Distance 0 (self): {num_clusters}")
                    print(f"    Distance 1: {path_lengths[1]} pairs ({path_lengths[1]/total_pairs*100:.2f}%)")
                    print(f"    Distance 2: {path_lengths[2]} pairs ({path_lengths[2]/total_pairs*100:.2f}%)")
                    print(f"    Distance 3: {path_lengths[3]} pairs ({path_lengths[3]/total_pairs*100:.2f}%)")

                    return edge_count, inserted_edges, adj_matrix, dist_matrix, \
                        last_edge_weight, min_edge_weight, max_edge_weight

                if edge_count % 50 == 0:
                    print(f"  {edge_count} edges: Connected, max distance = {max_dist}")
            else:
                if edge_count % 100 == 0:
                    print(f"  {edge_count} edges: Not yet connected")

    # 所有边都插入了
    print(f"\n⚠ All {edge_count} edges inserted")
    dist_matrix = bfs_shortest_paths(adj_matrix, num_clusters)
    min_edge_weight = min(e[2] for e in inserted_edges) if inserted_edges else 0
    max_edge_weight = max(e[2] for e in inserted_edges) if inserted_edges else 0

    return edge_count, inserted_edges, adj_matrix, dist_matrix, \
        last_edge_weight, min_edge_weight, max_edge_weight

if __name__ == "__main__":
    data = fvecs_read('/home/jlc/research/nsg-merge/data/sift/random/bi-index-data/sift_query.fvecs')
    q100 = data[:100,:]
    q1000 = data[:1000,:]
    print(q100.shape)
    print(q1000.shape)
    fvecs_write('/home/jlc/research/nsg-merge/data/sift/random/bi-index-data/sift_100query.fvecs', q100)
    fvecs_write('/home/jlc/research/nsg-merge/data/sift/random/bi-index-data/sift_1000query.fvecs', q1000)