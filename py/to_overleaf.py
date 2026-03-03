import csv
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import re
import os

# datasets = ['deep10m', 'anton10m', 'imagenet10m', 'msmarc10m']
datasets = ['sift']
methods = ['random']
merge_m = [2]
et = 0

if __name__=="__main__":
    base_path = ''
    csv_files = []
    line_names = []
    for db in datasets:
        print(f"Processing dataset: {db} ################################################")
        for method in methods:
            for m in merge_m:
                if m == 2:
                    base_path = f'/mnt/ssd/merge_bench/{db}/{method}/performance/bi/K10/'
                    output = f'/home/jlc/hnsw-merge/performance/bi'
                if m > 2:
                    base_path = f'/home/jlc/hnsw-merge/performance/{db}/{m}parts'
                    output = f'/home/jlc/hnsw-merge/performance'


                csv_files = glob.glob(f"{base_path}/*.csv")

                line_names = []
                for file in csv_files:
                    filename = os.path.basename(file)  # 获取文件名
                    match = re.search(fr'{db}_{method}_(.+)\.csv', filename)
                    if match:
                        line_names.append(match.group(1))


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

                        print(f"{csv_file}: ")
                        for recall, latency in zip(recalls, latencies):
                            print(f"({recall}, {latency})")
                    except Exception as e:
                        print(f"处理文件 {csv_file} 时出错: {e}")

                log_file = f'{base_path}/RGTM_merge.log'

                merge_times = []
                names = []

