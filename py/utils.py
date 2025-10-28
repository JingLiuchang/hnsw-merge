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


if __name__ == "__main__":
    dbs = ["sift", "deep1M", "glove100d", "crawl", "msong", "gist", "msmarco1M"]
    for db in dbs:
        data = fvecs_read(f'/mnt/ssd/merge_bench/{db}/{db}_base.fvecs')
        query = fvecs_read(f'/mnt/ssd/merge_bench/{db}/{db}_query.fvecs')
        print(f'{db} : {data.shape}; query : {query.shape}')
    # data = fvecs_read('/home/jlc/research/nsg-merge/data/sift/random/bi-index-data/sift_query.fvecs')
    # q100 = data[:100,:]
    # q1000 = data[:1000,:]
    # print(q100.shape)
    # print(q1000.shape)
    # fvecs_write('/home/jlc/research/nsg-merge/data/sift/random/bi-index-data/sift_100query.fvecs', q100)
    # fvecs_write('/home/jlc/research/nsg-merge/data/sift/random/bi-index-data/sift_1000query.fvecs', q1000)