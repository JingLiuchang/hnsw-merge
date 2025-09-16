//
// Created by jlc on 9/15/25.
//
#include "../../hnswlib/hnswlib.h"
#include <thread>
#include "../../hnswlib/utils.h"

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


int main(int argc, char** argv) {
    if (argc != 7) {
        std::cout << argv[0]
                  << " data_file ef_construction M graph_num graph_index_file merged_nsg_path"
                  << std::endl;
        exit(-1);
    }

    float* data = NULL;
    int max_elements, dim;
    load_data(argv[1], data, max_elements, dim);
    int ef_construction = atoi(argv[2]);
    int M = atoi(argv[3]);
    int graph_num = atoi(argv[4]);
    std::string graph_index_file = std::string(argv[5]);
    std::string merged_nsg_path = std::string(argv[6]);

    int num_threads = 72;       // Number of threads for operations with index

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::MergeHierarchicalNSW<float>* alg_hnsw = new hnswlib::MergeHierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    std::vector<hnswlib::HierarchicalNSW<float>*> graphs(graph_num);
    for (unsigned i = 0; i < graph_num; i++)
    {
        std::string index_file = graph_index_file + std::to_string(i+1) + "_ef50_M16.hnsw";
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* hnsw = new hnswlib::HierarchicalNSW<float>(&space);
        hnsw->loadIndex(index_file, &space);
        graphs[i] = hnsw;
    }

    alg_hnsw->merge(graph_num, graphs);

    alg_hnsw->saveIndex(merged_nsg_path);


    delete[] data;
    delete alg_hnsw;
    return 0;
}
