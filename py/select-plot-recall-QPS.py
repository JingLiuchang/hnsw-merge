import csv
import numpy as np
import matplotlib.pyplot as plt
import re

# 数据路径和文件名
base_path = f'/home/jlc/hnswlib/data/deep1M/kmeans/performance/5parts'

def plot(filename, name, color):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头（如果有的话）
        recalls = []
        latencies = []
        for row in reader:
            L, recall, latency = map(float, row)
            recalls.append(recall)
            latencies.append(latency)
        sorted_data = sorted(zip(recalls, latencies))
        recalls = [data[0] for data in sorted_data]
        latencies = [data[1] for data in sorted_data]
        plt.plot(recalls, latencies, marker='o', linestyle='-', label=name, color=color)

csv_files = [
    'deep1M_kmeans_BuildAsOne_ef80_M32.csv',
    'deep1M_kmeans_NGM_et0_ef80_M32.csv',
    'deep1M_kmeans_NGM_et0_ef20_M32.csv',
    'deep1M_kmeans_RGTM_et0_ef80_10_3_M32.csv',
    'deep1M_kmeans_RGTM_et0_ef80_10_5_M32.csv',
    'deep1M_kmeans_RGTM_et0_ef40_20_5_M32.csv',
    'deep1M_kmeans_RGTM_et0_ef40_20_3_M32.csv',
    'deep1M_kmeans_RGTM_et0_ef20_10_5_M32.csv',
    'deep1M_kmeans_RGTM_et0_ef20_10_3_M32.csv',
    'deep1M_kmeans_RGTM_et0_ef20_5_3_M32.csv'
]

line_names = ['BuildAsOne_ef80_M32', 'NGM_et0_ef80_M32', 'NGM_et0_ef20_M32', 'RGTM_et0_ef80_10_3_M32', 'RGTM_et0_ef80_10_5_M32', 'RGTM_et0_ef40_20_5_M32', 'RGTM_et0_ef40_20_3_M32', 'RGTM_et0_ef20_10_5_M32', 'RGTM_et0_ef20_10_3_M32', 'RGTM_et0_ef20_5_3_M32']


# csv_files = [
#     'deep1M_random_BuildAsOne_ef80_M32.csv',
#     'deep1M_random_NGM_et0_ef80_M32.csv',
#     'deep1M_random_NGM_et0_ef20_M32.csv',
#     'deep1M_random_RGTM_et0_ef80_10_5_M32.csv',
#     'deep1M_random_RGTM_et0_ef80_30_5_M32.csv',
#     'deep1M_random_RGTM_et0_ef40_20_5_M32.csv',
#     'deep1M_random_RGTM_et0_ef20_10_5_M32.csv'
# ]
#
# line_names = ['BuildAsOne_ef80_M32', 'NGM_et0_ef80_M32', 'NGM_et0_ef20_M32',
#               'RGTM_et0_ef80_10_5_M32', 'RGTM_et0_ef80_30_5_M32',
#               'RGTM_et0_ef40_20_5_M32', 'RGTM_et0_ef20_10_5_M32']

# 为每个line_name定义固定颜色
colors = plt.cm.tab10(np.linspace(0, 1, len(line_names)))  # 使用tab10配色方案
color_map = {name: colors[i] for i, name in enumerate(line_names)}

if __name__ == "__main__":
    # 第一张图：Recall vs Latency
    plt.figure(figsize=(10, 6))
    for i, csv_file in enumerate(csv_files):
        file_path = f"{base_path}/{csv_file}"
        recalls = []
        latencies = []

        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头（如果有的话）
                for row in reader:
                    L, recall, latency = map(float, row)
                    recalls.append(recall)
                    latencies.append(latency)

            # 按照recall值排序，确保图像平滑
            sorted_data = sorted(zip(recalls, latencies))
            recalls = [data[0] for data in sorted_data]
            latencies = [data[1] for data in sorted_data]

            # 使用固定颜色
            plt.plot(recalls, latencies, marker='o', linestyle='-', label=line_names[i], color=color_map[line_names[i]])
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    plot('/home/jlc/hnswlib/data/deep1M/overlap/performance/5parts/deep1M_overlap_ef80_M32.csv', 'overlap_ef80_M32', 'black')
    plot('/home/jlc/hnswlib/data/deep1M/overlap/performance/5parts/deep1M_overlap_ef40_M32.csv', 'overlap_ef40_M32', 'gray')

    targets = line_names
    targets = targets + ['overlap_ef80_M32', 'overlap_ef40_M32']
    color_map['overlap_ef80_M32'] = 'black'
    color_map['overlap_ef40_M32'] = 'gray'


    plt.xlabel('Recall')
    plt.ylabel('QPS')
    plt.title('Recall vs Latency for Different Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 可选：对y轴使用对数刻度，因为延迟可能有很大差异
    # plt.yscale('log')

    plt.tight_layout()
    plt.savefig(f"{base_path}/selected_recall_vs_QPS_comparison.png", dpi=300)
    plt.show()

    # 第二张图：Merge Time
    log_file = f'{base_path}/RGTM_merge.log'
    merge_times = []
    names = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Merge time'):
                # 提取Merge time值
                merge_time_pattern = r"Merge time:\s*([\d.]+)\s*s"
                merge_time_match = re.search(merge_time_pattern, line)
                merge_time = float(merge_time_match.group(1)) if merge_time_match else None

                # 提取deep1M_method_X.hnsw中的X部分
                hnsw_pattern = fr"deep1M_kmeans_(.+)\.hnsw"
                hnsw_match = re.search(hnsw_pattern, line)
                x_part = hnsw_match.group(1) if hnsw_match else None

                if not x_part:
                    hnsw_pattern = fr"deep1M_(.+)\.hnsw"
                    hnsw_match = re.search(hnsw_pattern, line)
                    x_part = hnsw_match.group(1) if hnsw_match else None

                if x_part not in targets:
                    continue

                merge_times.append(merge_time)
                names.append(x_part)


    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, merge_times, color=[color_map[name] for name in names])  # 使用固定颜色
    plt.xlabel('Configuration')
    plt.ylabel('Merge Time (s)')
    plt.title('Merge Time Comparison')
    plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
    plt.tight_layout()
    plt.savefig(f'{base_path}/selected-merge_time_comparison.png')
    plt.show()