
import numpy as np

import pandas as pd

import struct

# Set the font to Noto Sans CJK
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
# plt.rcParams['axes.unicode_minus'] = False  # Ensure that the minus sign is displayed correctly

# -*- coding: UTF-8 -*-
"""
Python script to read SIFT1B dataset files, including .bvecs, .ivecs, .fvecs
Dataset source: http://corpus-texmex.irisa.fr/
"""
import numpy as np


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

def ivecs_write_nocp(fname, m):
    n, d = m.shape
    with open(fname, 'wb') as f:
        for row in m:
            # Write the dimension first
            f.write(np.array([d], dtype='int32').tobytes())
            # Write the vector
            f.write(row.astype('int32').tobytes())

def fvecs_write_nocp(fname, m):
    n, d = m.shape
    with open(fname, 'wb') as f:
        for row in m:
            # Write the dimension first
            f.write(np.array([d], dtype='int32').tobytes())
            # Write the vector as float32
            f.write(row.astype('float32').tobytes())

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


def read_fbin_mmap(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file using memory mapping
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        # Read header
        header = np.fromfile(f, count=2, dtype=np.int32)
        total_nvecs, dim = header

        # Calculate sizes and offset
        header_size = 8  # 2 int32 numbers (8 bytes)
        vector_size = dim * 4  # float32 = 4 bytes
        offset = header_size + start_idx * vector_size

        # Determine number of vectors to read
        if chunk_size is None:
            nvecs = total_nvecs - start_idx
        else:
            nvecs = min(chunk_size, total_nvecs - start_idx)

        # Create memory mapping
        mm = np.memmap(filename, dtype=np.float32, mode='r',
                       offset=offset,
                       shape=(nvecs, dim))

        return mm

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
    if vecs.ndim != 2 or vecs.dtype != np.uint8:
        raise ValueError("must be 2-dim uint8 array")

    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape

        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('uint8').flatten().tofile(f)

def write_u8bin_nocp(filename: str, vecs: np.ndarray, ) -> None:
    if vecs.ndim != 2 or vecs.dtype != np.uint8:
        raise ValueError("must be 2-dim uint8 array")

    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape

        print(f'nvecs={nvecs} dim={dim}')

        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('uint8', copy=False).tofile(f)




