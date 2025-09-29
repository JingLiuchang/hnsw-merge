import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import re
import os

datasets = ['deep1M']
methods = ['random']
merge_m = [5]
et = 0

if __name__=="__main__":
    base_path = ''
    for db in datasets:
        for method in methods:
            for m in merge_m:
                if m == 2:
                    base_path = f'/home/jlc/hnswlib/data/{db}/{method}/performance/bi'
                if m > 2:
                    base_path = f'/home/jlc/hnswlib/data/{db}/{method}/performance/{m}parts'

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

                            # 提取deep1M_random_X.hnsw中的X部分
                            hnsw_pattern = fr"deep1M_{method}_(.+)\.hnsw"
                            hnsw_match = re.search(hnsw_pattern, line)
                            x_part = hnsw_match.group(1) if hnsw_match else None

                            merge_times.append(merge_time)
                            names.append(x_part)

                plt.figure(figsize=(12, 6))
                plt.bar(names, merge_times)
                plt.xlabel('Configuration')
                plt.ylabel('Merge Time (s)')
                plt.title('Merge Time Comparison')
                plt.xticks(rotation=45, ha='right')  # 旋转x轴标签，避免重叠
                plt.tight_layout()
                plt.savefig(f'{base_path}/merge_time_comparison.png')




