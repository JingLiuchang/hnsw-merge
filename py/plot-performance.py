import csv
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import re
import os

# datasets = ['anton10m', 'imagenet10m', 'deep10m', 'msmarc10m']
datasets = ['deep10m']
methods = ['kmeans']
merge_m = [20]
et = 0

if __name__=="__main__":
    base_path = ''
    csv_files = []
    line_names = []
    for db in datasets:
        for method in methods:
            for m in merge_m:
                if m == 2:
                    base_path = f'/mnt/ssd/merge_bench/{db}/{method}/performance/bi'
                    output = f'/home/jlc/hnsw-merge/performance/bi'
                if m > 2:
                    base_path = f'/home/jlc/hnsw-merge/performance/{db}/{m}parts'
                    output = f'/home/jlc/hnsw-merge/performance'


                csv_files = glob.glob(f"{base_path}/*.csv")

                line_names = []
                for file in csv_files:
                    filename = os.path.basename(file)
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
                            next(reader)
                            for row in reader:
                                L, recall, latency = map(float, row)
                                recalls.append(recall)
                                latencies.append(latency)

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

                if not exists(output):
                    os.makedirs(output)
                plt.tight_layout()
                plt.savefig(f"{base_path}/{db}_recall_vs_QPS_comparison.png", dpi=300)
                plt.show()