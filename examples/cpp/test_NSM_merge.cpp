//
// Created for NSM (Neighbor Sliding Merge) algorithm testing
//
#include "../../hnswlib/hnswlib.h"
#include <thread>
#include "../../hnswlib/utils.h"
#include "omp.h"
#include "../../hnswlib/parameter.h"

// Multithreaded executor
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

        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
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
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


int main(int argc, char** argv) {
    if (argc < 12 || argc > 14) {
        std::cout << argv[0]
                  << " data_file global_ef local_ef M sub_ef sub_M graph_num graph_index_file merged_nsg_path k_plus merge_order_selection [T] [merge_order_file]"
                  << std::endl;
        exit(-1);
    }

    float* data = NULL;
    int max_elements, dim;
    load_data(argv[1], data, max_elements, dim);
    int ef_construction = atoi(argv[2]);
    int local_ef = atoi(argv[3]);
    int M = atoi(argv[4]);
    int sub_ef = atoi(argv[5]);
    int sub_M = atoi(argv[6]);
    int graph_num = atoi(argv[7]);
    std::string graph_index_file = std::string(argv[8]);
    std::string merged_nsg_path = std::string(argv[9]);
    int k_plus = atoi(argv[10]);
    std::string merge_order = std::string(argv[11]);
    std::string merge_order_file = "None";
    int T = 1;  // 默认值为1
    if (argc == 13) {
        T = atoi(argv[12]);
    } else if (argc == 14) {
        T = atoi(argv[12]);
        merge_order_file = std::string(argv[13]);
    }

    hnswlib::Parameters params;
    params.Set<int>("local_ef", local_ef);
    params.Set<int>("k_plus", k_plus);
    params.Set<std::string>("method", "NSM_optimized");
    params.Set<std::string>("merge_order_selection", merge_order);
    params.Set<std::string>("merge_order_file", merge_order_file);
    params.Set<bool>("print", false);

    if (params.Get<bool>("print")) {
        std::cout << "parameters: " << std::endl;
        std::cout << "global_ef: " << ef_construction << std::endl;
        std::cout << "local_ef: " << local_ef << std::endl;
        std::cout << "M: " << M << std::endl;
        std::cout << "sub_ef: " << sub_ef << std::endl;
        std::cout << "sub_M: " << sub_M << std::endl;
        std::cout << "graph_num: " << graph_num << std::endl;
        std::cout << "graph_index_file: " << graph_index_file << std::endl;
        std::cout << "merged_nsg_path: " << merged_nsg_path << std::endl;
        std::cout << "k_plus: " << k_plus << std::endl;
        std::cout << "merge_order_selection: " << merge_order << std::endl;
        std::cout << "merge_order_file: " << merge_order_file << std::endl;
        std::cout << "T: " << T << std::endl;
        std::cout << "print: " << std::endl;
        std::cout << std::endl;
    }


    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::MergeHierarchicalNSW<float>* alg_hnsw = new hnswlib::MergeHierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    std::vector<hnswlib::HierarchicalNSW<float>*> graphs(graph_num);
    for (unsigned i = 0; i < graph_num; i++)
    {
        std::string index_file = graph_index_file + std::to_string(i+1) + "_ef" + std::to_string(sub_ef) + "_M" + std::to_string(sub_M) + ".hnsw";
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space);
        hnsw->loadIndex(index_file, &space);
        graphs[i] = hnsw;
    }

    int num_threads = T;
    omp_set_num_threads(num_threads);

    auto s = std::chrono::high_resolution_clock::now();
    alg_hnsw->mgraph_merge(graph_num, graphs, params);
    auto e = std::chrono::high_resolution_clock::now();

    double merge_time = std::chrono::duration<double>(e - s).count();

    std::cout << "Merge time: " << merge_time << " s; " << merged_nsg_path.substr(merged_nsg_path.find_last_of('/') + 1) << std::endl;

    alg_hnsw->saveIndex(merged_nsg_path);

    delete[] data;
    delete alg_hnsw;
    return 0;
}
