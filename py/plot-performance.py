import csv
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import re
import os


datasets = ['deep10m']
# datasets = ['deep1M', 'sift', 'msmarco1M', 'msong', 'anton1m', 'imagenet1m']
methods = ['random']
merge_m = [2]
et = 0

if __name__=="__main__":
    base_path = ''
    csv_files = []
    line_names = []
    for db in datasets:
        for method in methods:
            for m in merge_m:
                if m == 2:
                    base_path = f'/mnt/ssd/merge_bench/{db}/{method}/performance/bi/K10'
                    output = f'/home/jlc/hnsw-merge/performance/independent-vs-adjacent'
                if m > 2:
                    base_path = f'/mnt/ssd/merge_bench/{db}/{method}/performance/{m}parts/K100'
                    output = f'/home/jlc/hnsw-merge/performance/{m}parts'


                csv_files = glob.glob(f"{base_path}/*.csv")

                line_names = []
                for file in csv_files:
                    filename = os.path.basename(file)  # 获取文件名
                    match = re.search(fr'{db}_{method}_(.+)\.csv', filename)
                    if match:
                        line_names.append(match.group(1))


                plt.figure(figsize=(10, 6))

                num_files = len(csv_files)
                colors = cm.tab10(np.linspace(0, 1, num_files)) if num_files <= 10 else cm.Set3(np.linspace(0, 1, num_files))

                for i, csv_file in enumerate(csv_files):
                    file_path = csv_file
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

                        plt.plot(recalls, latencies, marker='o', linestyle='-', label=line_names[i], color=colors[i])
                    except Exception as e:
                        print(f"处理文件 {csv_file} 时出错: {e}")

                plt.xlabel('Recall')
                plt.ylabel('QPS')
                plt.title('Recall vs QPS for Different Algorithms')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()

                # 可选：对y轴使用对数刻度，因为延迟可能有很大差异
                # plt.yscale('log', base=2)

                if not exists(output):
                    os.makedirs(output)
                plt.tight_layout()
                plt.savefig(f"{output}/{db}_recall_vs_QPS_comparison_K10.png", dpi=300)
                plt.show()