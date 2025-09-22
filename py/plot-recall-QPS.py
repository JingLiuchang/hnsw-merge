import csv
import numpy as np
import matplotlib.pyplot as plt

# csv格式:L,recall,latency
base_path = f'/home/jlc/hnswlib/data/deep1M/random/performance/bi'

csv_files = ['deep1M_random_BuildAsOne_ef80_M32.csv', 'deep1M_random_NGM_et0_ef80_M32.csv', 'deep1M_random_RGTM_et0_ef80_10_3_M32.csv']
line_names = ['BuildAsOne', 'NGM_et0', 'RGTM_et0_ef80_10_3']

if __name__=="__main__":
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

            plt.plot(recalls, latencies, marker='o', linestyle='-', label=line_names[i])
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

    plt.xlabel('Recall')
    plt.ylabel('QPS')
    plt.title('Recall vs Latency for Different Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # 可选：对y轴使用对数刻度，因为延迟可能有很大差异
    # plt.yscale('log')

    plt.tight_layout()
    plt.savefig(f"{base_path}/recall_vs_QPS_comparison.png", dpi=300)
    plt.show()