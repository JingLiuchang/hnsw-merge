import numpy as np
from collections import deque
import faiss
import utils
import argparse

S = 4.0

def build_two_hop_graph(C: np.ndarray, R: int) -> np.ndarray:
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    assert C.shape == (n, n)
    assert np.allclose(C, C.T), "C must be symmetric"
    assert R >= 0 and isinstance(R, (int, np.integer)), "R must be a non-negative integer"

    # ---------- construct order ----------
    order_scores = []
    for i in range(n):
        d = np.delete(C[i], i)
        k = min(R, n - 1)
        if k == 0:
            score = 0.0
        else:
            score = np.partition(d, k - 1)[:k].mean()
        order_scores.append((score, i))
    order = [i for _, i in sorted(order_scores, key=lambda x: (x[0], x[1]))]

    # ---------- init ----------
    S = np.zeros_like(C)
    deg = np.zeros(n, dtype=int)
    E = 0

    # min distance matrix
    INF = 10**9
    dist = np.full((n, n), INF, dtype=int)
    for i in range(n):
        dist[i, i] = 0

    # adjacency list
    nbrs = [set() for _ in range(n)]

    def add_edge(u, v):
        nonlocal E
        if u == v or v in nbrs[u]:
            return False
        nbrs[u].add(v)
        nbrs[v].add(u)
        S[u, v] = S[v, u] = C[u, v]
        deg[u] += 1
        deg[v] += 1
        E += 1
        # incrementally update dist matrix
        bfs_update(u)
        bfs_update(v)
        return True

    def bfs_update(src):
        # using BFS to update dist[src,*]
        q = deque()

        visited = set()
        q.append(src)
        visited.add(src)
        dist[src, src] = 0
        while q:
            x = q.popleft()
            dx = dist[src, x]
            for y in nbrs[x]:
                if dist[src, y] > dx + 1:
                    dist[src, y] = dx + 1
                    dist[y, src] = dx + 1
                    if y not in visited:
                        visited.add(y)
                        q.append(y)

    target_edges = (n * R + 1) // 2  # target number of edges

    # ---------- main ----------
    for v in order:
        # early stop
        if E >= target_edges:
            break

        # any node at distance >= 3 from v
        far_nodes = [x for x in range(n) if x != v and dist[v, x] >= 3]
        far_nodes.sort(key=lambda x: C[v, x])

        for x in far_nodes:
            if E >= target_edges:
                break
            if v == x or x in nbrs[v]:
                continue

            v_cap = deg[v] < R
            x_cap = deg[x] < R

            if v_cap and x_cap:
                add_edge(v, x)
                continue

            # 1-hop relay
            candidate_edge_done = False
            if v_cap and not x_cap:
                candidates = [y for y in nbrs[x] if deg[y] < R and y != v and y not in nbrs[v]]
                if candidates:
                    y = min(candidates, key=lambda y: (dist[v, y], C[v, y], y))
                    add_edge(v, y)
                    candidate_edge_done = True

            # 1-hop relay (symmetric)
            if not candidate_edge_done and not v_cap and x_cap:
                candidates = [y for y in nbrs[v] if deg[y] < R and y != x and x not in nbrs[y]]
                if candidates:
                    y = min(candidates, key=lambda y: (dist[x, y], C[x, y], y))
                    add_edge(x, y)
                    candidate_edge_done = True

            if not candidate_edge_done:
                add_edge(v, x)

    np.fill_diagonal(S, 0.0)
    return S

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="K-means clustering for dataset partitioning.")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of clusters for K-means.")
    parser.add_argument("--db", type=str, required=True, help="Dataset name (e.g., 'sift').")
    parser.add_argument("--R", type=int, default=-1, help="Degree limit for the 2-hop graph.")
    args = parser.parse_args()

    num_clusters = args.num_clusters
    db = args.db
    R = args.R

    if R <= 0:
        tm = np.ceil(S * np.ceil(np.sqrt(num_clusters)))
        R = int(tm + 1)

    if num_clusters <= 2:
        print('Error: num_clusters must be at least 3.')
        exit(1)

    # Load centroid data
    kmeans_centroids_path = f'/mnt/ssd/merge_bench/{db}/kmeans/multi-index-data/{num_clusters}parts/{db}_kmeans_centroids.fvecs'
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

    print(pairwise_distances)

    spanner = build_two_hop_graph(pairwise_distances, R)
    print(spanner)
    print("2-hop graph avg degree:", np.count_nonzero(spanner) / N)

    utils.fvecs_write(f'/mnt/ssd/merge_bench/{db}/kmeans/multi-index-data/{num_clusters}parts/{db}_centroid_2hopgraph.fvecs', spanner)