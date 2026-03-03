//
// Created by jlc on 9/26/25.
//
#include "../../hnswlib/hnswlib.h"
#include <thread>
#include "../../hnswlib/utils.h"
#include "omp.h"
#include "../../hnswlib/parameter.h"
#include "../../hnswlib/space_l2.h"

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] { // 创造该线程后，该线程自己执行下面的内容，而上面的循环开始创造下一个线程，主线程只负责创建线程，并将线程对象存储到 threads 容器中
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join(); // 让主线程（或调用 join() 的线程）阻塞，直到被 join() 的线程完成运行
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

// template<typename dist_t>
// void shard_data_into_clusters(float* data,
//     float* centroids,
//     int data_num,
//     int centroid_num,
//     int dim,
//     unsigned kbase,
//     hnswlib::SpaceInterface<dist_t> *s,
//     std::vector<std::tuple<hnswlib::labeltype, unsigned, hnswlib::labeltype>>& assignments_list,
//     std::vector<std::vector<hnswlib::labeltype>>& idmaps)
// {
//     hnswlib::DISTFUNC<dist_t> fstdistfunc_ = s->get_dist_func();
//     void *dist_func_param_ = s->get_dist_func_param();
//
//     // Assignments for each data point
//     std::vector<std::vector<unsigned>> assignments(data_num);
//
// #pragma omp parallel for schedule(dynamic)
//     for (int i = 0; i < data_num; i++)
//     {
//         std::vector<std::pair<dist_t, unsigned>> distances;
//
//         // Compute the distance from data[i] to each centroid
//         for (unsigned j = 0; j < centroid_num; j++)
//         {
//             dist_t distance = fstdistfunc_(data + i * dim, centroids + j * dim, dist_func_param_);
//             distances.emplace_back(distance, j);
//         }
//
//         // Sort the distances to find the kbase nearest centroids
//         std::partial_sort(distances.begin(), distances.begin() + kbase, distances.end(),
//             [](const std::pair<dist_t, unsigned>& a, const std::pair<dist_t, unsigned>& b) {
//                 return a.first < b.first; // Sort by distance (ascending)
//             });
//
//         // Extract the kbase nearest centroid IDs
//         std::vector<unsigned> assignment;
//         for (unsigned k = 0; k < kbase; k++)
//         {
//             assignment.push_back(distances[k].second);
//         }
//
//         // Store the assignment for the current data point
//         assignments[i] = std::move(assignment);
//     }
//
//     // Initialize idmaps and counts for each centroid
//     idmaps.resize(centroid_num);
//     std::vector<int> centroid_counts(centroid_num, 0);
//
//     // Count how many data points are assigned to each centroid
//     for (size_t i = 0; i < data_num; i++)
//     {
//         for (unsigned k = 0; k < kbase; k++)
//         {
//             unsigned cid = assignments[i][k];
//             centroid_counts[cid]++;
//         }
//     }
//
//     // Resize idmaps based on the counts
//     for (unsigned cid = 0; cid < centroid_num; cid++)
//     {
//         idmaps[cid].resize(centroid_counts[cid]);
//         centroid_counts[cid] = 0; // Reset counts to reuse as indices
//     }
//
//     // Populate assignments_list and idmaps
//     for (size_t i = 0; i < data_num; i++)
//     {
//         for (unsigned k = 0; k < kbase; k++)
//         {
//             unsigned cid = assignments[i][k];
//             unsigned local_id = centroid_counts[cid]++;
//
//             // Add to assignments_list
//             assignments_list.emplace_back(i, cid, local_id);
//
//             // Update idmaps
//             idmaps[cid][local_id] = i;
//         }
//     }
//
//     // At this point, idmaps and assignments_list are constructed
//     // If needed, centroid-wise data sets can also be created
//     for (unsigned cid = 0; cid < centroid_num; cid++)
//     {
//         int num_points = centroid_counts[cid];
//         if (num_points > 0)
//         {
//             // Create a dataset for this centroid
//             float* centroid_data = new float[num_points * dim];
//
//             // Populate dataset
//             for (int local_id = 0; local_id < num_points; local_id++)
//             {
//                 int global_id = idmaps[cid][local_id];
//                 std::copy(data + global_id * dim, data + (global_id + 1) * dim, centroid_data + local_id * dim);
//             }
//
//             // The `centroid_data` can be used for further processing.
//             // NOTE: Remember to delete it later when no longer needed.
//             delete[] centroid_data;
//         }
//     }
// }

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cout << argv[0]
                  << " data_file ef_construction M graph_num subdata_file subgraph_index_file merged_nsg_path"
                  << std::endl;
        exit(-1);
    }

    float* data = NULL;
    int max_elements, dim;
    load_data(argv[1], data, max_elements, dim);
    int ef_construction = atoi(argv[2]);
    int M = atoi(argv[3]);
    int graph_num = atoi(argv[4]);
    std::string subdata_file = std::string(argv[5]);
    std::string subgraph_index_file = std::string(argv[6]);
    std::string merged_nsg_path = std::string(argv[7]);
    unsigned kbase = 2;
    std::string centroid_file = subdata_file + "_kmeans_centroids.fvecs";
    std::string idmap_file = subdata_file + "_idmaps_kbase2.bin";
    std::string assignment_file = subdata_file + "_kbase2_assignments.bin";
    std::string global2local_file = subdata_file + "_global2local_kbase2.bin";

    float* centroid_data = NULL;
    int centroid_num, centroid_dim;
    load_data(const_cast<char*>(centroid_file.c_str()), centroid_data, centroid_num, centroid_dim);

    std::vector<std::pair<hnswlib::labeltype, unsigned>> assignments;
    std::vector<std::vector<hnswlib::labeltype>> idmaps;
    std::vector<std::unordered_map<unsigned, size_t>> global_to_local_map;
    read_assignments(assignment_file, assignments);
    read_idmaps(idmap_file, idmaps);
    loadGlobalToLocalMapFromBinary(global2local_file, global_to_local_map);

    hnswlib::Parameters params;
    params.Set<bool>("print", true);
    params.Set<unsigned>("kbase", kbase);

    int num_threads = 8;       // Number of threads for operations with index
    omp_set_num_threads(num_threads);
    double time_cost = 0.0;

    hnswlib::L2Space space(dim);
    // shard_data_into_clusters(data, centroid_data, max_elements, centroid_num, dim,2, &space, assignments, idmaps); // shard data into clusters according to centroids

    // build sub graphs
    std::vector<hnswlib::HierarchicalNSW<float>*> graphs(graph_num);
    for (unsigned i = 0; i < graph_num; i++)
    {
        // std::string index_file = subgraph_index_file + std::to_string(i+1) + "_ef" + std::to_string(ef_construction) + "_M" + std::to_string(M) + ".hnsw";

        std::string data_file = subdata_file + "_centroid_" + std::to_string(i+1) + ".fvecs";
        float* subdata;
        int sub_max_elements, sub_dim;
        safe_load_data(const_cast<char*>(data_file.c_str()), subdata, sub_max_elements, sub_dim);

        hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space, sub_max_elements, M, ef_construction);

        auto s = std::chrono::high_resolution_clock::now();
        ParallelFor(0, sub_max_elements, num_threads, [&](size_t row, size_t threadId) {
            hnsw->addPoint((void*)(subdata + sub_dim * row), row); // fn(row, threadId)在threadId号线程中执行alg_hnsw->addPoint, row: partition id
        });
        auto e = std::chrono::high_resolution_clock::now();
        time_cost += std::chrono::duration<double>(e - s).count();

        graphs[i] = hnsw;
    }
    std::cout << "Build sub-graphs time: " << time_cost << " s" << std::endl;

    // Initing merged index
    hnswlib::MergeHierarchicalNSW<float>* alg_hnsw = new hnswlib::MergeHierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    auto s = std::chrono::high_resolution_clock::now();
    alg_hnsw->overlap_merge(data, dim, graph_num, graphs, assignments, idmaps, global_to_local_map, params);
    auto e = std::chrono::high_resolution_clock::now();

    double merge_time = std::chrono::duration<double>(e - s).count();
    time_cost += merge_time;

    std::cout << "Merge shards time: " << merge_time << " s" << std::endl;
    std::cout << "Merge time: " << time_cost << " s; " << merged_nsg_path.substr(merged_nsg_path.find_last_of('/') + 1) << std::endl;

    alg_hnsw->saveIndex(merged_nsg_path);

    delete[] data;
    delete alg_hnsw;
    return 0;
}

// int main()
// {
//     std::string idmap_file = "/home/jlc/hnswlib/data/sift/overlap/multi-index-data/5parts/sift_idmaps_k2.bin";
//     std::string assignment_file = "/home/jlc/hnswlib/data/sift/overlap/multi-index-data/5parts/sift_kbase2_assignments.bin";
//
//     std::vector<std::pair<int, int>> assignments;
//     std::vector<std::vector<int>> idmaps;
//
//     read_assignments(assignment_file, assignments);
//     read_idmaps(idmap_file, idmaps);
//
//     return 0;
// }