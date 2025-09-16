//
// Created by jlc on 9/15/25.
//
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
    if (argc != 9) {
        std::cout << argv[0]
                  << "data_file query_file gt_file graph_index_path k min_ef max_ef stepsize"
                  << std::endl;
        exit(-1);
    }

    float* data = NULL;
    int max_elements, dim;
    load_data(argv[1], data, max_elements, dim);

    float* query = NULL;
    int query_elements, query_dim;
    load_data(argv[2], query, query_elements, query_dim);

    // auto dist_gt = read_fvecs(argv[1]);
    std::vector<std::vector<unsigned>> gt = read_ivecs(argv[3]);

    std::string graph_index_path = std::string(argv[4]);
    int k = atoi(argv[5]);
    int min_ef = atoi(argv[6]);
    int max_ef = atoi(argv[7]);
    int stepsize = atoi(argv[8]);

    int num_threads = 1;       // Number of threads for operations with index

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, graph_index_path);

    std::cout << "ef " << "Recall@" << k << " " << "QPS " << std::endl;
    for (int ef = min_ef; ef <= max_ef; ef += stepsize) {
        alg_hnsw->setEf(ef);

        std::vector<std::vector<hnswlib::labeltype>> neighbors(query_elements);
        auto s = std::chrono::high_resolution_clock::now();
        ParallelFor(0, query_elements, num_threads, [&](size_t row, size_t threadId) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(query + query_dim * row, k);
            for (int i = 0; i < k; i++) {
                hnswlib::labeltype label = result.top().second;
                neighbors[row].push_back(label);
                result.pop();
            }
        });
        auto e = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double>(e - s).count();
        double QPS = query_elements / latency;

        std::vector<double> recalls;
        double recall = compute_recall(neighbors, gt, recalls);

        std::cout << ef << " " << recall << " " << QPS << std::endl;
    }

    delete[] data;
    delete alg_hnsw;
    return 0;
}
