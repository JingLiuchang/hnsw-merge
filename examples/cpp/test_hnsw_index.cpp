//
// Created by jlc on 9/15/25.
//
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
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}


int main(int argc, char** argv) {
    if (argc != 5) {
        std::cout << argv[0]
                  << "data_file ef_construction M graph_index_path"
                  << std::endl;
        exit(-1);
    }

    float* data = NULL;
    int max_elements, dim;
    safe_load_data(argv[1], data, max_elements, dim);
    int ef_construction = atoi(argv[2]);
    int M = atoi(argv[3]);
    std::string graph_index_path = std::string(argv[4]);

    int num_threads = 118;       // Number of threads for operations with index

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    auto s = std::chrono::high_resolution_clock::now();
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)(data + dim * row), row);
    });
    auto e = std::chrono::high_resolution_clock::now();

    double index_time = std::chrono::duration<double>(e - s).count();
    std::cout << "Index time: " << index_time << " s; " << graph_index_path.substr(graph_index_path.find_last_of('/') + 1) << std::endl;

    alg_hnsw->saveIndex(graph_index_path);

    delete[] data;
    delete alg_hnsw;
    return 0;
}
