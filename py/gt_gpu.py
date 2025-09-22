import numpy as np
import torch
import time
import os
import struct
import argparse
import utils

def read_vectors(filename, max_vectors=None):
    """通用向量读取函数，支持fvecs/fbin格式"""
    ext = os.path.splitext(filename)[1].lower()

    if ext == '.fbin':
        with open(filename, 'rb') as f:
            nvec = np.frombuffer(f.read(4), dtype=np.int32)[0]
            dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            data = np.fromfile(f, dtype=np.float32)
            return data.reshape(nvec, dim)
    elif ext == '.fvecs':
        with open(filename, 'rb') as f:
            vecs = []
            while True:
                header = f.read(4)
                if not header: break
                dim = np.frombuffer(header, dtype=np.int32)[0]
                vec = np.fromfile(f, count=dim, dtype=np.float32)
                vecs.append(vec)
            return np.stack(vecs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def process_data(base_file, query_file, chunk_size=1000000, top_k=100):
    # 读取查询数据并验证维度
    queries = read_vectors(query_file)
    Q = torch.from_numpy(queries).cuda().float()
    query_dim = Q.shape[1]
    q_norms = (Q ** 2).sum(dim=1)

    # 初始化结果存储
    top_distances = torch.full((len(queries), top_k), float('inf'), device='cuda')
    top_indices = torch.full((len(queries), top_k), -1, dtype=torch.long, device='cuda')

    # 获取基础数据信息
    base_ext = os.path.splitext(base_file)[1].lower()
    if base_ext == '.fbin':
        with open(base_file, 'rb') as f:
            nvec = np.frombuffer(f.read(4), dtype=np.int32)[0]
            base_dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            data_offset = 8
            bytes_per_vector = base_dim * 4
    elif base_ext == '.fvecs':
        with open(base_file, 'rb') as f:
            first_header = f.read(4)
            base_dim = np.frombuffer(first_header, dtype=np.int32)[0]
            data_offset = 0
            bytes_per_vector = (base_dim + 1) * 4  # 4字节头 + 数据
    else:
        raise ValueError(f"Unsupported base file format: {base_ext}")

    # 验证维度一致性
    if base_dim != query_dim:
        raise ValueError(f"Dimension mismatch: base has {base_dim}D, query has {query_dim}D")

    # 分块处理逻辑
    with open(base_file, 'rb') as f:
        f.seek(data_offset)
        offset = 0
        chunk_idx = 0

        while True:
            start_time = time.time()
            raw_data = f.read(chunk_size * bytes_per_vector)
            if not raw_data: break

            # 处理不完整数据块
            valid_bytes = len(raw_data) // bytes_per_vector * bytes_per_vector
            raw_data = raw_data[:valid_bytes]
            num_vectors = len(raw_data) // bytes_per_vector

            # 解析数据块
            if base_ext == '.fbin':
                vectors = np.frombuffer(raw_data, dtype=np.float32).reshape(num_vectors, base_dim)
            else:
                data = np.frombuffer(raw_data, dtype=np.uint8)
                data = data.reshape(num_vectors, bytes_per_vector)
                vectors = data[:, 4:].view(np.float32).reshape(num_vectors, base_dim)

            # GPU计算
            B = torch.from_numpy(vectors).cuda().float()
            b_norms = (B ** 2).sum(dim=1)
            dists = q_norms.unsqueeze(1) + b_norms - 2 * (Q @ B.T)

            # 合并结果
            current_dists, current_idx = torch.topk(dists, k=top_k, dim=1, largest=False)
            current_idx += offset

            merged_dists = torch.cat([top_distances, current_dists], 1)
            merged_idx = torch.cat([top_indices, current_idx], 1)

            top_dists, top_idx = torch.topk(merged_dists, k=top_k, dim=1, largest=False)
            top_indices = torch.gather(merged_idx, 1, top_idx)
            top_distances = top_dists

            offset += num_vectors
            chunk_idx += 1

            print(f'Chunk {chunk_idx}: {num_vectors} vectors processed in {time.time()-start_time:.2f}s')

    return top_indices.cpu().numpy(), top_distances.cpu().numpy()

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

def process_pairwise_data(base_file, query_file, chunk_size=1000000, top_k=100, query_batch_size=1000):
    # 读取查询数据并验证维度
    queries = read_vectors(query_file)
    query_dim = queries.shape[1]
    num_queries = queries.shape[0]

    # 初始化结果存储
    top_distances = torch.full((num_queries, top_k), float('inf'))
    top_indices = torch.full((num_queries, top_k), -1, dtype=torch.long)

    # 获取基础数据信息
    base_ext = os.path.splitext(base_file)[1].lower()
    if base_ext == '.fbin':
        with open(base_file, 'rb') as f:
            nvec = np.frombuffer(f.read(4), dtype=np.int32)[0]
            base_dim = np.frombuffer(f.read(4), dtype=np.int32)[0]
            data_offset = 8
            bytes_per_vector = base_dim * 4
    elif base_ext == '.fvecs':
        with open(base_file, 'rb') as f:
            first_header = f.read(4)
            base_dim = np.frombuffer(first_header, dtype=np.int32)[0]
            data_offset = 0
            bytes_per_vector = (base_dim + 1) * 4  # 4字节头 + 数据
    else:
        raise ValueError(f"Unsupported base file format: {base_ext}")

    # 验证维度一致性
    if base_dim != query_dim:
        raise ValueError(f"Dimension mismatch: base has {base_dim}D, query has {query_dim}D")

    # 处理查询批次
    for q_start in range(0, num_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, num_queries)
        q_batch = queries[q_start:q_end]
        Q = torch.from_numpy(q_batch).cuda().float()
        q_norms = (Q ** 2).sum(dim=1, keepdim=True)

        # 初始化当前批次的结果
        batch_top_distances = torch.full((q_end - q_start, top_k), float('inf'), device='cuda')
        batch_top_indices = torch.full((q_end - q_start, top_k), -1, dtype=torch.long, device='cuda')

        # 分块处理基础数据
        with open(base_file, 'rb') as f:
            f.seek(data_offset)
            offset = 0
            chunk_idx = 0

            while True:
                start_time = time.time()
                raw_data = f.read(chunk_size * bytes_per_vector)
                if not raw_data: break

                # 处理不完整数据块
                valid_bytes = len(raw_data) // bytes_per_vector * bytes_per_vector
                raw_data = raw_data[:valid_bytes]
                num_vectors = len(raw_data) // bytes_per_vector

                # 解析数据块
                if base_ext == '.fbin':
                    vectors = np.frombuffer(raw_data, dtype=np.float32).reshape(num_vectors, base_dim)
                else:
                    data = np.frombuffer(raw_data, dtype=np.uint8)
                    data = data.reshape(num_vectors, bytes_per_vector)
                    vectors = data[:, 4:].view(np.float32).reshape(num_vectors, base_dim)

                # GPU计算
                B = torch.from_numpy(vectors).cuda().float()
                b_norms = (B ** 2).sum(dim=1, keepdim=True)
                dists = q_norms + b_norms.t() - 2 * (Q @ B.T)

                # 合并结果
                current_dists, current_idx = torch.topk(dists, k=min(top_k, num_vectors), dim=1, largest=False)
                current_idx += offset

                # 合并批次结果
                merged_dists = torch.cat([batch_top_distances, current_dists], 1)
                merged_idx = torch.cat([batch_top_indices, current_idx], 1)

                batch_top_dists, top_idx = torch.topk(merged_dists, k=top_k, dim=1, largest=False)
                batch_top_indices = torch.gather(merged_idx, 1, top_idx)
                batch_top_distances = batch_top_dists

                offset += num_vectors
                chunk_idx += 1

                print(f'Query batch {q_start//query_batch_size+1}, Chunk {chunk_idx}: {num_vectors} vectors processed in {time.time()-start_time:.2f}s')

        # 将当前批次结果复制到CPU
        top_distances[q_start:q_end] = batch_top_distances.cpu()
        top_indices[q_start:q_end] = batch_top_indices.cpu()

        print(f'Processed query batch {q_start//query_batch_size+1}/{(num_queries-1)//query_batch_size+1}')

        # 清理GPU内存
        torch.cuda.empty_cache()

    return top_indices.numpy(), top_distances.numpy()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Random partitioning for dataset partitioning.")
    parser.add_argument("--base_file", type=str, required=True)
    parser.add_argument("--query_file", type=str, required=True)
    parser.add_argument("--gt_file", type=str, required=True)
    parser.add_argument("--dist_file", type=str, required=True)
    args = parser.parse_args()

    base_file = args.base_file
    query_file = args.query_file
    gt_file = args.gt_file
    dist_file = args.dist_file

    indices, distances = process_pairwise_data(base_file, query_file,100000,100)

    print("shape of indices:", indices.shape)
    print("shape of distances:", distances.shape)
    print(indices[0][:10])
    print(distances[0][:10])

    utils.ivecs_write(gt_file, indices)
    utils.fvecs_write(dist_file, distances)