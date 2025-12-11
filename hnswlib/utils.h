//
// Created by jlc on 9/15/25.
//

#include <fstream>
#include <iostream>
#include <sys/stat.h>
#ifndef UTILS_H
#define UTILS_H

void load_data(char* filename, float*& data, size_t& num,
               size_t& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    int dim_int;
    in.read((char*)&dim_int, 4);
    dim = (size_t)dim_int;
    //std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (size_t)(fsize / (dim + 1) / 4);
    data = new float[num * dim];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}
// void load_data(char* filename, float*& data, int& num, int& dim) {
//     std::ifstream in(filename, std::ios::binary);
//     if (!in.is_open()) {
//         std::cerr << "open file error" << std::endl;
//         exit(-1);
//     }
//
//     in.read((char*)&dim, 4);
//     if (!in || dim <= 0) {
//         std::cerr << "Invalid dimension value: " << dim << std::endl;
//         exit(-1);
//     }
//
//     in.seekg(0, std::ios::end);
//     size_t filesize = (size_t)in.tellg();
//     if (filesize < 4) {
//         std::cerr << "File is too small to contain valid data" << std::endl;
//         exit(-1);
//     }
//     num = (int)((filesize - 4) / (dim * 4 + 4));
//     data = new (std::nothrow) float[num * dim];
//     if (!data) {
//         std::cerr << "Failed to allocate memory for data" << std::endl;
//         exit(-1);
//     }
//
//     in.seekg(4, std::ios::beg);
//     for (int i = 0; i < num; ++i) {
//         in.seekg(4, std::ios::cur);
//         in.read((char*)(data + i * dim), dim * 4);
//         if (!in) {
//             std::cerr << "Failed to read data at index " << i << std::endl;
//             delete[] data;
//             exit(-1);
//         }
//     }
//
//     in.close();
// }

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

    float* current_ptr = data;
    for (int i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)current_ptr, dim * 4);
        current_ptr += dim;

        if (in.fail()) {
            std::cout << "Read failed at vector " << i << std::endl;
            delete[] data;
            data = nullptr;
            return;
        }
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
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        if (fin.eof()) {
            break;
        }

        std::vector<unsigned> vec(dim);
        fin.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(unsigned));

        if (!fin) {
            break;
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
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        if (fin.eof()) {
            break;
        }

        std::vector<float> vec(dim);
        fin.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));

        if (!fin) {
            break;
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
        csv_file.open(file_path, std::ios::app);
    }
    else if (exists and !add)
    {
        csv_file.open(file_path, std::ios::trunc);
        csv_file << "L,recall,QPS" << std::endl;
    }
    else if (!exists)
    {
        csv_file.open(file_path);
        csv_file << "L,recall,QPS" << std::endl;
    }

    if (!csv_file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return;
    }

    csv_file << L << "," << recall << "," << QPS << std::endl;

    csv_file.close();
}

bool read_assignments(const std::string& filename, std::vector<std::pair<size_t, unsigned int>>& assignments) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open assignments file: " << filename << std::endl;
        return false;
    }

    unsigned int num_assignments;
    file.read(reinterpret_cast<char*>(&num_assignments), sizeof(unsigned int));
    if (file.fail()) {
        std::cerr << "Error: Failed to read number of assignments" << std::endl;
        file.close();
        return false;
    }

    std::cout << "Reading " << num_assignments << " assignments..." << std::endl;

    assignments.reserve(num_assignments);
    assignments.clear();

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

bool read_idmaps(const std::string& filename, std::vector<std::vector<size_t>>& idmaps) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open idmaps file: " << filename << std::endl;
        return false;
    }

    unsigned int num_centroids;
    file.read(reinterpret_cast<char*>(&num_centroids), sizeof(unsigned int));
    if (file.fail()) {
        std::cerr << "Error: Failed to read number of centroids" << std::endl;
        file.close();
        return false;
    }

    std::cout << "Reading idmaps for " << num_centroids << " centroids..." << std::endl;

    idmaps.clear();
    idmaps.resize(num_centroids);

    for (unsigned int cid = 0; cid < num_centroids; cid++) {
        size_t num_points;
        file.read(reinterpret_cast<char*>(&num_points), sizeof(size_t));
        if (file.fail()) {
            std::cerr << "Error: Failed to read number of points for centroid " << cid << std::endl;
            file.close();
            return false;
        }

        idmaps[cid].reserve(num_points);
        idmaps[cid].clear();

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


size_t get_global_id(const std::vector<std::vector<size_t>>& idmaps, unsigned int centroid_id, size_t local_id) {
    if (centroid_id >= idmaps.size()) {
        std::cerr << "Error: Invalid centroid_id: " << centroid_id << std::endl;
        return SIZE_MAX;
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

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }


    size_t num_data = 0;
    file.read(reinterpret_cast<char*>(&num_data), sizeof(size_t));


    global_to_local_map.clear();
    global_to_local_map.resize(num_data);


    for (size_t global_id = 0; global_id < num_data; ++global_id) {

        unsigned map_size = 0;
        file.read(reinterpret_cast<char*>(&map_size), sizeof(unsigned));


        for (unsigned i = 0; i < map_size; ++i) {
            unsigned cid = 0;
            size_t local_id = 0;

            file.read(reinterpret_cast<char*>(&cid), sizeof(unsigned));
            file.read(reinterpret_cast<char*>(&local_id), sizeof(size_t));


            global_to_local_map[global_id][cid] = local_id;
        }
    }
}

#endif //UTILS_H
