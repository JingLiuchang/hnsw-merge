//
// Created by jlc on 9/15/25.
//

#include <fstream>
#include <iostream>
#include <sys/stat.h>

#ifndef UTILS_H
#define UTILS_H

void load_data(char* filename, float*& data, int& num,
               int& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    //std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (int)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

std::vector<std::vector<unsigned>> read_ivecs(const std::string& filename) {
    std::vector<std::vector<unsigned>> vectors;
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open " + filename);
    }

    while (true) {
        int32_t dim;
        // 读取向量维度
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        if (fin.eof()) {
            break;
        }

        // 读取向量数据
        std::vector<unsigned> vec(dim);
        fin.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(unsigned));

        if (!fin) {
            break; // 如果读取失败则退出
        }

        vectors.push_back(std::move(vec));
    }

    return vectors;
}

std::vector<std::vector<float>> read_fvecs(const std::string& filename) {
    std::vector<std::vector<float>> vectors;
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open " + filename);
    }

    while (true) {
        int32_t dim;
        // 读取向量维度
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        if (fin.eof()) {
            break;
        }

        // 读取向量数据
        std::vector<float> vec(dim);
        fin.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));

        if (!fin) {
            break; // 如果读取失败则退出
        }

        vectors.push_back(std::move(vec));
    }

    return vectors;
}

double compute_recall(const std::vector<std::vector<hnswlib::labeltype>>& res,
                      const std::vector<std::vector<unsigned>>& gt,
                      std::vector<double>& recalls) {
    // Check for invalid input sizes
    if (res.empty() || gt.empty() || res.size() != gt.size()) {
        throw std::runtime_error("Invalid input sizes");
    }

    double total_recall = 0.0;
    const size_t query_num = res.size(); // Number of queries
    const size_t K = res[0].size(); // Assume all ground truth vectors have the same size

    for (size_t i = 0; i < query_num; i++) {
        // Convert ground truth to a set for fast lookup
        std::unordered_set<unsigned> gt_set(gt[i].begin(), gt[i].begin() + K);

        // Count the number of hits (elements in res[i] that are also in gt_set)
        size_t hits = 0;
        for (const auto& id : res[i]) {
            if (gt_set.count(static_cast<unsigned>(id)) > 0) { // Ensure type compatibility
                hits++;
            }
        }

        // Compute recall for the current query
        double recall = static_cast<double>(hits) / K;
        recalls.push_back(recall);
        total_recall += recall;
    }

    // Return average recall over all queries
    return total_recall / query_num;
}

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}

void write_csv_data(const std::string& file_path, int L, double recall, double QPS, bool add) {
    bool exists = file_exists(file_path);

    std::ofstream csv_file;

    if (exists and add)
    {
        // 追加模式打开文件
        csv_file.open(file_path, std::ios::app);
    }
    else if (exists and !add)
    {
        // 覆盖模式打开文件
        csv_file.open(file_path, std::ios::trunc);
        // 写入表头
        csv_file << "L,recall,QPS" << std::endl;
    }
    else if (!exists)
    {
        // 创建新文件
        csv_file.open(file_path);
        // 写入表头
        csv_file << "L,recall,QPS" << std::endl;
    }

    if (!csv_file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    // 写入数据行
    csv_file << L << "," << recall << "," << QPS << std::endl;

    csv_file.close();
}

#endif //UTILS_H
