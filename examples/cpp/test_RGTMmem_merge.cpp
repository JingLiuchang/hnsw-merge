#include "../../hnswlib/hnswlib.h"
#include "../../hnswlib/parameter.h"
#include "../../hnswlib/utils.h"
#include "peak_rss_monitor.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "omp.h"

int main(int argc, char** argv) {
    if (argc < 14 || argc > 16) {
        std::cout << argv[0]
                  << " data_file global_ef local_ef self_ef M sub_ef sub_M graph_num graph_index_file merged_nsg_path ET ratio merge_order_selection [T] [merge_order_file]"
                  << std::endl;
        return -1;
    }

    int max_elements = 0;
    int dim = 0;
    load_data_metadata(argv[1], max_elements, dim);
    int global_ef = std::atoi(argv[2]);
    int local_ef = std::atoi(argv[3]);
    int self_ef = std::atoi(argv[4]);
    int M = std::atoi(argv[5]);
    int sub_ef = std::atoi(argv[6]);
    int sub_M = std::atoi(argv[7]);
    int graph_num = std::atoi(argv[8]);
    std::string graph_index_file = argv[9];
    std::string merged_nsg_path = argv[10];
    int early_terminate = std::atoi(argv[11]);
    float ratio = std::atof(argv[12]);
    std::string merge_order = argv[13];
    std::string merge_order_file = "None";
    int num_threads = 1;
    if (argc == 15) {
        num_threads = std::atoi(argv[14]);
    } else if (argc == 16) {
        num_threads = std::atoi(argv[14]);
        merge_order_file = argv[15];
    }
    if (graph_num <= 0 || num_threads <= 0) {
        std::cerr << "graph_num and T must be positive" << std::endl;
        return -1;
    }

    hnswlib::Parameters params;
    params.Set<bool>("early_terminate", early_terminate);
    params.Set<float>("et_ratio", ratio);
    params.Set<int>("local_ef", local_ef);
    params.Set<int>("self_ef", self_ef);
    params.Set<std::string>("method", "RGTMmem");
    params.Set<std::string>("merge_order_selection", merge_order);
    params.Set<std::string>("merge_order_file", merge_order_file);
    params.Set<bool>("print", true);

    hnswlib::L2Space space(dim);
    std::vector<std::unique_ptr<hnswlib::HierarchicalNSW<float>>> owned_graphs;
    std::vector<hnswlib::HierarchicalNSW<float>*> graphs;
    owned_graphs.reserve(graph_num);
    graphs.reserve(graph_num);
    for (int graph_id = 0; graph_id < graph_num; ++graph_id) {
        std::string index_file = graph_index_file + std::to_string(graph_id + 1) +
                                 "_ef" + std::to_string(sub_ef) +
                                 "_M" + std::to_string(sub_M) + ".hnsw";
        owned_graphs.emplace_back(new hnswlib::HierarchicalNSW<float>(&space));
        owned_graphs.back()->loadIndex(index_file, &space);
        graphs.push_back(owned_graphs.back().get());
    }

    std::unique_ptr<hnswlib::MergeHierarchicalNSW<float>> merged_index(
        new hnswlib::MergeHierarchicalNSW<float>(
            &space, max_elements, M, global_ef, hnswlib::GraphOnlyStorageTag()));

    omp_set_num_threads(num_threads);
    auto start = std::chrono::high_resolution_clock::now();
    PeakRssMonitor memory_monitor;
    merged_index->mgraph_merge(graph_num, graphs, params);
    const double core_peak_rss_gb = memory_monitor.stopGb();
    auto end = std::chrono::high_resolution_clock::now();

    double merge_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Merge time: " << merge_time << " s; "
              << merged_nsg_path.substr(merged_nsg_path.find_last_of('/') + 1) << std::endl;
    printPeakRss("RGTMmem core peak RSS", core_peak_rss_gb);

    merged_index->saveIndex(merged_nsg_path);
    return 0;
}
