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
    data = new float[num * dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

void safe_load_data(char* filename, float*& data, int& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }

    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (int)(fsize / (dim + 1) / 4);

    size_t total_elements = (size_t)num * (size_t)dim;
    data = new float[total_elements];

    in.seekg(0, std::ios::beg);

    // 使用指针递增而不是索引计算
    float* current_ptr = data;
    for (int i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)current_ptr, dim * 4);
        current_ptr += dim;  // 指针递增，避免大数乘法

        if (in.fail()) {
            std::cout << "Read failed at vector " << i << std::endl;
            delete[] data;
            data = nullptr;
            return;
        }
    }

    in.close();
}

void load_data_metadata(const char* filename, int& num, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(std::string("Cannot open data file: ") + filename);
    }

    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim <= 0) {
        throw std::runtime_error(std::string("Invalid fvecs header: ") + filename);
    }

    in.seekg(0, std::ios::end);
    std::streamoff file_size = in.tellg();
    const std::streamoff record_size =
        static_cast<std::streamoff>(sizeof(int32_t)) +
        static_cast<std::streamoff>(dim) * static_cast<std::streamoff>(sizeof(float));
    if (file_size <= 0 || file_size % record_size != 0) {
        throw std::runtime_error(std::string("Invalid fvecs file size: ") + filename);
    }
    num = static_cast<int>(file_size / record_size);
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

void write_ndc_csv_data(const std::string& file_path, int L, double NDC, double recall, bool add) {
    bool exists = file_exists(file_path);

    std::ofstream csv_file;

    if (exists && add) {
        csv_file.open(file_path, std::ios::app);
    } else if (exists && !add) {
        csv_file.open(file_path, std::ios::trunc);
        csv_file << "L,NDC,recall" << std::endl;
    } else {
        csv_file.open(file_path);
        csv_file << "L,NDC,recall" << std::endl;
    }

    if (!csv_file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    csv_file << L << "," << NDC << "," << recall << std::endl;
    csv_file.close();
}

/**
 * 读取assignments文件
 * @param filename assignments文件路径
 * @param assignments 输出的assignments向量，pair<size_t, unsigned int>
 * @return 成功返回true，失败返回false
 */
bool read_assignments(const std::string& filename, std::vector<std::pair<size_t, unsigned int>>& assignments) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open assignments file: " << filename << std::endl;
        return false;
    }

    // 读取assignments数量 (unsigned int)
    unsigned int num_assignments;
    file.read(reinterpret_cast<char*>(&num_assignments), sizeof(unsigned int));
    if (file.fail()) {
        std::cerr << "Error: Failed to read number of assignments" << std::endl;
        file.close();
        return false;
    }

    std::cout << "Reading " << num_assignments << " assignments..." << std::endl;

    // 预分配内存
    assignments.reserve(num_assignments);
    assignments.clear();

    // 读取每个assignment
    for (unsigned int i = 0; i < num_assignments; i++) {
        size_t data_id;
        unsigned int centroid_id;

        file.read(reinterpret_cast<char*>(&data_id), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&centroid_id), sizeof(unsigned int));

        if (file.fail()) {
            std::cerr << "Error: Failed to read assignment " << i << std::endl;
            file.close();
            return false;
        }

        assignments.emplace_back(data_id, centroid_id);
    }

    file.close();
    std::cout << "Successfully read " << assignments.size() << " assignments" << std::endl;
    return true;
}

/**
 * 读取idmaps文件
 * @param filename idmaps文件路径
 * @param idmaps 输出的idmaps向量，idmaps[centroid_id][local_id] = global_id
 * @return 成功返回true，失败返回false
 */
bool read_idmaps(const std::string& filename, std::vector<std::vector<size_t>>& idmaps) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open idmaps file: " << filename << std::endl;
        return false;
    }

    // 读取centroids数量 (unsigned int)
    unsigned int num_centroids;
    file.read(reinterpret_cast<char*>(&num_centroids), sizeof(unsigned int));
    if (file.fail()) {
        std::cerr << "Error: Failed to read number of centroids" << std::endl;
        file.close();
        return false;
    }

    std::cout << "Reading idmaps for " << num_centroids << " centroids..." << std::endl;

    // 初始化idmaps
    idmaps.clear();
    idmaps.resize(num_centroids);

    // 读取每个centroid的idmap
    for (unsigned int cid = 0; cid < num_centroids; cid++) {
        // 读取当前centroid的数据点数量 (size_t)
        size_t num_points;
        file.read(reinterpret_cast<char*>(&num_points), sizeof(size_t));
        if (file.fail()) {
            std::cerr << "Error: Failed to read number of points for centroid " << cid << std::endl;
            file.close();
            return false;
        }

        // 预分配内存
        idmaps[cid].reserve(num_points);
        idmaps[cid].clear();

        // 读取每个local_id对应的global_id (size_t)
        for (size_t local_id = 0; local_id < num_points; local_id++) {
            size_t global_id;
            file.read(reinterpret_cast<char*>(&global_id), sizeof(size_t));
            if (file.fail()) {
                std::cerr << "Error: Failed to read global_id for centroid " << cid
                         << ", local_id " << local_id << std::endl;
                file.close();
                return false;
            }

            idmaps[cid].push_back(global_id);
        }

        std::cout << "Centroid " << cid << ": " << idmaps[cid].size() << " points" << std::endl;
    }

    file.close();
    std::cout << "Successfully read idmaps for all centroids" << std::endl;
    return true;
}

/**
 * 根据centroid_id和local_id获取global_id
 * @param idmaps idmaps向量
 * @param centroid_id 聚类中心ID
 * @param local_id 局部ID
 * @return global_id，如果索引无效返回SIZE_MAX
 */
size_t get_global_id(const std::vector<std::vector<size_t>>& idmaps, unsigned int centroid_id, size_t local_id) {
    if (centroid_id >= idmaps.size()) {
        std::cerr << "Error: Invalid centroid_id: " << centroid_id << std::endl;
        return SIZE_MAX;  // 使用SIZE_MAX表示错误
    }

    if (local_id >= idmaps[centroid_id].size()) {
        std::cerr << "Error: Invalid local_id: " << local_id
                 << " for centroid: " << centroid_id << std::endl;
        return SIZE_MAX;
    }

    return idmaps[centroid_id][local_id];
}

void loadGlobalToLocalMapFromBinary(const std::string& filename,
                                    std::vector<std::unordered_map<unsigned, size_t>>& global_to_local_map) {
    // 打开二进制文件
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // 读取总大小（size_t，表示 global_to_local_map 的大小）
    size_t num_data = 0;
    file.read(reinterpret_cast<char*>(&num_data), sizeof(size_t));

    // 预分配空间
    global_to_local_map.clear();
    global_to_local_map.resize(num_data);

    // 读取每个 global_id 的数据
    for (size_t global_id = 0; global_id < num_data; ++global_id) {
        // 读取字典大小（unsigned，表示键值对数量）
        unsigned map_size = 0;
        file.read(reinterpret_cast<char*>(&map_size), sizeof(unsigned));

        // 读取 map_size 个键值对
        for (unsigned i = 0; i < map_size; ++i) {
            unsigned cid = 0;     // 读取 cid
            size_t local_id = 0; // 读取 local_id

            file.read(reinterpret_cast<char*>(&cid), sizeof(unsigned));
            file.read(reinterpret_cast<char*>(&local_id), sizeof(size_t));

            // 插入到 global_to_local_map 中
            global_to_local_map[global_id][cid] = local_id;
        }
    }
}

#endif //UTILS_H
