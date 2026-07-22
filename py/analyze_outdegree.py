"""
Analyze out-degree distribution of an hnswlib binary index file.

Usage:
    python analyze_outdegree.py <index_file> [--save <output_prefix>]

Output:
    - Per-level out-degree statistics (stdout)
    - Histogram plots saved as <output_prefix>_level<L>.png  (if --save given)
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Binary parsing
# ---------------------------------------------------------------------------

def read_header(f):
    """Parse the hnswlib index header. Returns a dict of header fields."""
    # Field types mirror saveIndex / loadIndex in hnswalg.h
    # size_t -> Q (uint64), int -> i (int32), unsigned int -> I (uint32), double -> d
    fields = [
        ("offsetLevel0",         "Q"),
        ("max_elements",         "Q"),
        ("cur_element_count",    "Q"),
        ("size_data_per_element","Q"),
        ("label_offset",         "Q"),
        ("offsetData",           "Q"),
        ("maxlevel",             "i"),
        ("enterpoint_node",      "I"),
        ("maxM",                 "Q"),
        ("maxM0",                "Q"),
        ("M",                    "Q"),
        ("mult",                 "d"),
        ("ef_construction",      "Q"),
    ]
    hdr = {}
    for name, fmt in fields:
        size = struct.calcsize(fmt)
        hdr[name] = struct.unpack(fmt, f.read(size))[0]
    return hdr


def read_index(path):
    """
    Read an hnswlib index file and return (header, level0_degrees, upper_degrees).

    level0_degrees : np.ndarray of shape (N,)  – out-degree of each node at level 0
    upper_degrees  : list of np.ndarray        – upper_degrees[l] = out-degrees at level l+1
                     (empty list if maxlevel == 0)
    """
    with open(path, "rb") as f:
        hdr = read_header(f)

        N              = hdr["cur_element_count"]
        sde            = hdr["size_data_per_element"]   # bytes per element in level-0 block
        maxM0          = hdr["maxM0"]
        maxM           = hdr["maxM"]
        maxlevel       = hdr["maxlevel"]

        size_links_level0    = maxM0 * 4 + 4   # uint32 count + uint32 * maxM0
        size_links_per_level = maxM  * 4 + 4   # uint32 count + uint32 * maxM  (upper layers)

        # ---- Level-0 block ------------------------------------------------
        level0_block_start = f.tell()
        level0_degrees = np.empty(N, dtype=np.int32)

        for i in range(N):
            # The very first uint32 in each element slot is the neighbor count
            (cnt,) = struct.unpack_from("I", f.read(sde))
            level0_degrees[i] = cnt

        # ---- Upper-level link lists ----------------------------------------
        # upper_counts[l] collects out-degree of each node at level l+1
        upper_counts = [[] for _ in range(maxlevel)]

        for i in range(N):
            (link_list_size,) = struct.unpack("I", f.read(4))
            if link_list_size == 0:
                # node lives only at level 0 – contribute 0 to all upper levels
                for l in range(maxlevel):
                    upper_counts[l].append(0)
            else:
                node_max_level = link_list_size // size_links_per_level
                raw = f.read(link_list_size)
                for l in range(maxlevel):
                    if l < node_max_level:
                        offset = l * size_links_per_level
                        (cnt,) = struct.unpack_from("I", raw, offset)
                        upper_counts[l].append(cnt)
                    else:
                        upper_counts[l].append(0)

    upper_degrees = [np.array(uc, dtype=np.int32) for uc in upper_counts]
    return hdr, level0_degrees, upper_degrees


# ---------------------------------------------------------------------------
# Analysis & plotting
# ---------------------------------------------------------------------------

def print_stats(label, degrees, max_cap):
    nonzero = degrees[degrees > 0]
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  nodes total       : {len(degrees):>10,}")
    print(f"  nodes with edges  : {len(nonzero):>10,}")
    print(f"  max capacity (M)  : {max_cap:>10}")
    if len(nonzero) == 0:
        print("  (no edges)")
        return
    print(f"  min degree        : {degrees.min():>10}")
    print(f"  max degree        : {degrees.max():>10}")
    print(f"  mean degree       : {degrees.mean():>10.3f}")
    print(f"  median degree     : {np.median(degrees):>10.1f}")
    print(f"  % nodes at max    : {100*(degrees == max_cap).mean():>9.2f}%")

    # Degree frequency table (compact)
    vals, cnts = np.unique(degrees, return_counts=True)
    print(f"\n  degree  count      pct")
    for v, c in zip(vals, cnts):
        bar = "#" * min(40, int(40 * c / len(degrees)))
        print(f"  {v:6d}  {c:8,}  {100*c/len(degrees):5.1f}%  {bar}")


def plot_histogram(degrees, label, max_cap, ax):
    bins = np.arange(0, max_cap + 2) - 0.5
    ax.hist(degrees, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(degrees.mean(), color="crimson", linestyle="--", linewidth=1.2,
               label=f"mean={degrees.mean():.1f}")
    ax.axvline(max_cap, color="orange", linestyle=":", linewidth=1.2,
               label=f"max_cap={max_cap}")
    ax.set_xlabel("Out-degree")
    ax.set_ylabel("Node count")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.5, max_cap + 1.5)


def analyze(index_path, save_prefix=None):
    print(f"\nReading index: {index_path}")
    hdr, l0_deg, upper_deg = read_index(index_path)

    print("\n--- Index header ---")
    for k, v in hdr.items():
        print(f"  {k:<25} = {v}")

    N        = hdr["cur_element_count"]
    maxlevel = hdr["maxlevel"]
    maxM0    = hdr["maxM0"]
    maxM     = hdr["maxM"]

    print_stats("Level 0", l0_deg, maxM0)
    for l, deg in enumerate(upper_deg):
        print_stats(f"Level {l+1}", deg, maxM)

    # ---- Plots ------------------------------------------------------------
    num_plots = 1 + len(upper_deg)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
    if num_plots == 1:
        axes = [axes]

    plot_histogram(l0_deg, f"Level 0  (maxM0={maxM0})", maxM0, axes[0])
    for l, deg in enumerate(upper_deg):
        plot_histogram(deg, f"Level {l+1}  (maxM={maxM})", maxM, axes[l + 1])

    plt.suptitle(f"Out-degree distribution\n{Path(index_path).name}  |  N={N:,}",
                 fontsize=11)
    plt.tight_layout()

    if save_prefix:
        out = f"{save_prefix}_outdegree.png"
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hnswlib index out-degree distribution")
    parser.add_argument("index_file", help="Path to .hnsw index file")
    parser.add_argument("--save", metavar="PREFIX",
                        help="Save plot to <PREFIX>_outdegree.png instead of displaying")
    args = parser.parse_args()

    analyze(args.index_file, save_prefix=args.save)
