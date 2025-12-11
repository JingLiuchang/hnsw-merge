//
// Created by jlc on 9/9/25.
//
#pragma once

#include <tbb/parallel_sort.h>
#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <memory>
#include "parameter.h"
#include <algorithm>
#include "utils.h"
#include <set>

#define S 4.0

void add_circulant_edges_k2(unsigned n,
                            unsigned t_max,            // 1..t_max
                            std::vector<std::pair<unsigned, unsigned>>& merge_order)
{
    for (unsigned t = 1; t <= t_max; t++) {
        for (unsigned i = 0; i < n; i++) {
            unsigned j = (i + t) % n;
            merge_order.emplace_back(i, j);
        }
    }

    if (n%2 == 0)
    {
        unsigned t = n/2;
        for (unsigned i = 0; i < n/2; i++) {
            unsigned j = (i + t) % n;
            merge_order.emplace_back(i, j);
        }
    }
    else
    {
        unsigned t1 = (n-1) / 2;
        unsigned t2 = t1 + 1;
        for (unsigned i = 0; i < t1; i++)
        {
            unsigned j1 = (i + t1) % n;
            unsigned j2 = (i + t2) % n;
            merge_order.emplace_back(i, j1);
            merge_order.emplace_back(i, j2);
        }
        unsigned last_i = t1;
        unsigned last_j = (last_i + t1) % n;
        merge_order.emplace_back(last_i, last_j);
    }

    std::set<std::pair<unsigned, unsigned>> unique_pairs;

    auto it = merge_order.begin();
    while (it != merge_order.end()) {
        unsigned i = it->first;
        unsigned j = it->second;

        // Check if i == j (self-loop) or if i or j are out of bounds
        if (i == j || i >= n || j >= n) {
            it = merge_order.erase(it); // Remove invalid pairs
            std::cerr << 'invalid pair removed: (' << i << ", " << j << ")\n";
            continue;
        }

        // Normalize the pair into (min(i, j), max(i, j))
        std::pair<unsigned, unsigned> normalized_pair = {std::min(i, j), std::max(i, j)};

        // Check if the normalized pair already exists
        if (unique_pairs.count(normalized_pair)) {
            it = merge_order.erase(it); // Remove duplicate (i, j) or (j, i)
        } else {
            unique_pairs.insert(normalized_pair); // Insert the pair into the set
            ++it;
        }
    }
}

void unweighted_graph_order(unsigned m, std::vector<std::pair<unsigned, unsigned>>& merge_order) // circulant order for random partition
{
    unsigned tc = 0;
    if (m%2 == 0)
    {
        tc = (m - 4 + 3) / 4;
    }
    else
    {
        tc = (m - 5 + 3) / 4;
    }

    unsigned tm = std::ceil((S * std::ceil(std::sqrt(m))));

    unsigned t = std::max(tc, tm) / 2;
    add_circulant_edges_k2(m, t, merge_order);
}

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;
typedef std::pair<labeltype, unsigned> localid_graphidtype;

template<typename dist_t>
class MergeHierarchicalNSW : public HierarchicalNSW<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0}; // maximum levels

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    // merge related
    std::unordered_map<mergeidtype, localid_graphidtype> mergeid_lookup_; // mergeid -> internal id , graph id
    std::vector<size_t> globalid_offset; // globalid_offset[i] = sum of elements in graphs before i-th graph


    MergeHierarchicalNSW(SpaceInterface<dist_t> *s){
    }


    MergeHierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted){
        loadIndex(location, s, max_elements);
    }


    MergeHierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : HierarchicalNSW<dist_t>(s, max_elements, M, ef_construction, random_seed),
            label_op_locks_(MAX_LABEL_OPERATION_LOCKS), // label: data + dim * label, the maximum threads: MAX_LABEL_OPERATION_LOCKS
            link_list_locks_(max_elements), // Each vector has a corresponding function to update its neighbor table. This function applies not only to the newly created vector itself but also to the involved neighbor nodes (especially with reverse edges).
            element_levels_(max_elements), // The highest-level layer ID for each vector, with the index being the internal ID.
            allow_replace_deleted_(allow_replace_deleted){
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint); // Neighbor ID array + Number of neighbors (Level 0)
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype); // Neighbor ID array + number of neighbors + vector data + label: Size of a complete data point in the graph (level 0).
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_); // Allocate maximum memory usage (Level 0)
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0; // inserted nodes number

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        // initializations for special treatment of the first node
        enterpoint_node_ = -1; // internal id
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: MergeHierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint); // The neighbor ID array plus the number of neighbors (not at level 0), and the size of an element in the linklist. The first 2 bytes of `linklistsizeint(unsigned int 4B)` represent the number of neighbors; the remaining bytes are related to adding or deleting elements.
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~MergeHierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };


    void setEf(size_t ef) {
        ef_ = ef;
    }


    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }


    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    // bare_bone_search means there is no check for deletions and stop condition is ignored in return of extra performance
    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search ||
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
//                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  ///////////
                                        _MM_HINT_T0);  ////////////////////////
#endif

                        if (bare_bone_search ||
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_); // The level is -1 because the linked list only stores non-zero levels.
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);//pruning
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true; // reverse edges exits
                        break;
                    }
                }
            }

            // When reverse edge does not exist, perform insertion
            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) { // not full, direct insert
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else { // full, conduct pruning
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }

    std::vector<tableint> Pruning_InterInsert(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level) {

        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0)
        {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        // outgoing edges
        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c]);
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            // if (*ll_cur && !isUpdate) {
            //     throw std::runtime_error("The newly inserted element should have blank link list");
            // }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                // if (data[idx] && !isUpdate)
                //     throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        // reverse edges
        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            for (size_t j = 0; j < sz_link_list_other; j++) {
                if (data[j] == cur_c) {
                    is_cur_c_present = true; // reverse edge already exists
                    break;
                }
            }


            // When reverse edge does not exist, perform insertion
            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) { // not full, direct insert
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else { // full, conduct pruning
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }
        return selectedNeighbors;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }


    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    *
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr); // nnbrs : unsigned short int type
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }


        // insert and delete related
        // check if there is vacant place
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0; // internal id
        { // update
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) { // the element with the same label already exists
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel; // highest level of cur_c

        std::unique_lock <std::mutex> templock(global); // locking maxlevel with global
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_; // init: -1
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_); // init cur_c's data (nnbrs, nids, vector, label)

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype)); // label = external id = partition id, write to cur_c's label
        memcpy(getDataByInternalId(cur_c), data_point, data_size_); // internal id: row number of level0's index layout

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) { // inserting to non-first point
            if (curlevel < maxlevelcopy) { // greedy search to curlevel's ep
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level); // currObj's nids at level
                        int size = getListCount(data); // nnbrs

                        tableint *datal = (tableint *) (data + 1); // +1: skipping 4B of linklistsizeint, linklistsizeint = tableint = unsigned int
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i]; // cand: neighbor's internal id
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            // insert to lower levels
            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false); // pruning and connect
            }
        } else { // the first element inserted, no ep
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c; // ep is always the element with highest level
            maxlevel_ = curlevel;
        }
        return cur_c;
    }


    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

    mergeidtype getGlobalidbyLocalid(labeltype local_id, unsigned graph_id)
    {
        return globalid_offset[graph_id] + local_id;
    }

    void initialize_mergeid_lookup(std::vector<HierarchicalNSW<dist_t>*> graphs) {
        size_t global_id_offset = 0;
        unsigned m = graphs.size();
        globalid_offset.resize(m);

        for (unsigned i = 0; i < m; ++i) {
            size_t size_i = graphs[i]->max_elements_;
            globalid_offset[i] = global_id_offset;

            for (size_t local_id = 0; local_id < size_i; ++local_id) { // local id and global id are labeltype
                mergeidtype global_id = global_id_offset + local_id;
                mergeid_lookup_[global_id] = std::make_pair(static_cast<tableint>(local_id), i);
            }
            global_id_offset += size_i;
        }
    }

    void pairwise_merge_order(unsigned m, std::vector<std::pair<unsigned, unsigned>>& merge_order) {
        merge_order.clear();
        for (unsigned i = 0; i < m; ++i) {
            for (unsigned j = i + 1; j < m; ++j) {
                merge_order.emplace_back(i, j);
            }
        }
    }

    void init_merge_graph_level0(std::vector<HierarchicalNSW<dist_t>*> graphs)
    {
        cur_element_count = max_elements_;

        maxlevel_ = std::numeric_limits<int>::min();

        for (unsigned i = 0; i < graphs.size(); ++i) {
            HierarchicalNSW<dist_t>* graph = graphs[i];
            if (graph->maxlevel_ > maxlevel_) {
                maxlevel_ = graph->maxlevel_;
                labeltype graph_ep = graph->getExternalLabel(graph->enterpoint_node_);
                enterpoint_node_ = getGlobalidbyLocalid(graph_ep, i);
            }
        }

        maxlevel_ = 0;

        size_t graph_offset = 0;
        for (unsigned graphid =0; graphid < graphs.size(); ++graphid)
        {
            auto graph = graphs[graphid];
            char* data_level0_memory_copy = data_level0_memory_ + graph_offset * size_data_per_element_ + offsetLevel0_;
            graph_offset += graph->cur_element_count;
            for (tableint id = 0; id < graph->cur_element_count; ++id)
            {
                char* copy_element_data = data_level0_memory_copy + id * size_data_per_element_;
                char* cur_element_data = graph->data_level0_memory_ + id * graph->size_data_per_element_ + graph->offsetLevel0_;

                memset(copy_element_data, 0, size_data_per_element_);

                // copying data
                memcpy(copy_element_data, cur_element_data, graph->size_links_level0_); // linklistsizeint 4B + maxM0 * sizeof(tableint) copy
                memcpy(copy_element_data + offsetData_, cur_element_data + graph->offsetData_, data_size_); // vector copy
                memcpy(copy_element_data + label_offset_, cur_element_data + graph->label_offset_, sizeof(labeltype)); // label copy

                // mapping id
                unsigned short int neighbor_count = getListCount((linklistsizeint*) copy_element_data);
                for (unsigned i = 0; i < neighbor_count; ++i)
                {
                    tableint internalid = *((tableint*)(cur_element_data + sizeof(linklistsizeint) + i * sizeof(tableint)));
                    tableint global_internalid = internalid + globalid_offset[graphid];
                    *(tableint*)(copy_element_data + sizeof(linklistsizeint) + i * sizeof(tableint)) = global_internalid;
                }
                labeltype local_id = *(labeltype *)(copy_element_data + label_offset_); // mapping label id
                mergeidtype global_id = getGlobalidbyLocalid(local_id, graphid);
                *(labeltype *)(copy_element_data + label_offset_) = global_id;
            }
        }
    }

    void mgraph_merge(unsigned m, std::vector<HierarchicalNSW<dist_t>*> graphs, const Parameters &parameters)
    {
        std::string method = parameters.Get<std::string>("method");
        std::string merge_order_selection = parameters.Get<std::string>("merge_order_selection");
        std::string merge_order_file = parameters.Get<std::string>("merge_order_file");

        size_t num_element = 0;
        for (unsigned i = 0; i < m; ++i)
        {
            num_element += graphs[i]->max_elements_;
        }

        if (num_element != max_elements_ or graphs.size() != m)
        {
            std::cerr << "Error: total number of elements or number of graphs does not match the initialized value." << std::endl;
            return;
        }

        initialize_mergeid_lookup(graphs);
        init_merge_graph_level0(graphs);

        // merge order selection
        std::vector<std::pair<unsigned, unsigned>> merge_order;
        if (merge_order_selection == "pairwise")
        {
            std::cout << "Pairwise merge order selected." << std::endl;
            pairwise_merge_order(m, merge_order);
        }
        else if (merge_order_selection == "mst")
        {
            std::cout << "MST merge order selected." << std::endl;
            if (merge_order_file == "None")
            {
                std::cerr << "Error: MST order file does not exist." << std::endl;
                exit(1);
            }
            std::vector<std::vector<float>> mst = read_fvecs(merge_order_file);
            for (unsigned i = 0; i < m; ++i)
            {
                for (unsigned j = 0; j < m; ++j) // TODO: bug here!!
                {
                    if (mst[i][j] != 0)
                    {
                        merge_order.emplace_back(i, j);
                    }
                }
            }
        }
        else if (merge_order_selection == "circle")
        {
            std::cout << "Circle merge order selected." << std::endl;
            std::vector<unsigned> nodes(m);

            for (unsigned i = 0; i < m; ++i) {
                nodes[i] = i;
            }

            unsigned seed = 42;
            std::mt19937 g(seed);
            std::shuffle(nodes.begin(), nodes.end(), g);

            for (unsigned i = 0; i < m; ++i) {
                merge_order.emplace_back(nodes[i], nodes[(i + 1) % m]);
            }

            merge_order.emplace_back(0,2);

            if (m == 6)
            {
                merge_order.emplace_back(0,2);
                merge_order.emplace_back(0,4);
                merge_order.emplace_back(1,3);
                merge_order.emplace_back(1,5);
                merge_order.emplace_back(2,4);
                merge_order.emplace_back(3,5);
            }

            if (m == 8)
            {
                merge_order.emplace_back(0,2);
                merge_order.emplace_back(1,3);
                merge_order.emplace_back(2,4);
                merge_order.emplace_back(3,5);
                merge_order.emplace_back(4,6);
                merge_order.emplace_back(5,7);
                merge_order.emplace_back(6,0);
                merge_order.emplace_back(7,1);
                merge_order.emplace_back(0,4);
                merge_order.emplace_back(1,5);
                merge_order.emplace_back(2,6);
                merge_order.emplace_back(3,7);
            }
        }
        else if (merge_order_selection == std::string("path"))
        {
            std::cout << "Path merge order selected." << std::endl;
            std::vector<unsigned> nodes(m);

            for (unsigned i = 0; i < m; ++i) {
                nodes[i] = i;
            }

            unsigned seed = 42;
            std::mt19937 g(seed);
            std::shuffle(nodes.begin(), nodes.end(), g);

            for (unsigned i = 0; i < m-1; ++i) {
                merge_order.emplace_back(nodes[i], nodes[(i + 1) % m]);
            }
        }
        else if (merge_order_selection == "weighted-graph")
        {
            std::cout << "Graph merge order selected." << std::endl;
            if (merge_order_file == "None")
            {
                std::cerr << "Error: graph order file does not exist." << std::endl;
                exit(1);
            }
            std::vector<std::vector<float>> G = read_fvecs(merge_order_file);
            for (unsigned i = 0; i < m; ++i)
            {
                for (unsigned j = i + 1; j < m; ++j)
                {
                    if (G[i][j] != 0)
                    {
                        merge_order.emplace_back(i, j);
                    }
                }
            }
        }
        else if (merge_order_selection == "unweighted-graph")
        {
            std::cout << "unweighted-graph merge order selected." << std::endl;

            unweighted_graph_order(m, merge_order);
        }
        else if (merge_order_selection == "pertersen")
        {
            std::cout << "Petersen merge order selected." << std::endl;
            if (m != 10)
            {
                std::cerr << "Error: pertersen merge order only supports 10 graphs." << std::endl;
                exit(1);
            }
            std::vector<unsigned> nodes(m);

            for (unsigned i = 0; i < m; ++i) {
                nodes[i] = i;
            }

            unsigned seed = 42;
            std::mt19937 g(seed);
            std::shuffle(nodes.begin(), nodes.end(), g);

            // Petersen graph
            std::vector<std::pair<unsigned, unsigned>> petersen_edges = {
                {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0,2}, {1,3}, {2,4}, {3,0}, {4,1}, {0, 6}, {1, 7}, {2, 8}, {3, 9}, {4, 5},
                {5, 7}, {7, 9}, {9, 6}, {6, 8}, {8, 5}, {5,6}, {6,7}, {7,8}, {8,9}, {9,5}, {5, 1}, {6, 2}, {7, 3}, {8, 4}, {9, 0},
                {0, 5}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
            };

            for (const auto& edge : petersen_edges) {
                unsigned u = nodes[edge.first];
                unsigned v = nodes[edge.second];
                merge_order.push_back({std::min(u, v), std::max(u, v)});
            }
        }
        else
        {
            std::cerr << "Error: unknown merge_order_selection " << merge_order_selection << std::endl;
            exit(1);
        }

        // start merging
        if (method == "NGM")
        {
            for (auto&& p : merge_order) { // pairwise merge
                NGM_merge_into_later(graphs, p.first, p.second, ef_construction_, parameters);
            }
        }
        else if (method == "RGTM")
        {
            for (auto&& p : merge_order) { // pairwise merge
                RGTM_merge_into_later(graphs, p.first, p.second, parameters);
            }
        }
        else
        {
            std::cerr << "Error: unknown method " << method << std::endl;
        }
    }

    void overlap_merge(float* data, int dim, unsigned m,
        std::vector<HierarchicalNSW<dist_t>*> graphs,
        const std::vector<std::pair<labeltype, unsigned>>& assignments,
        std::vector<std::vector<labeltype>>& idmaps,
        std::vector<std::unordered_map<unsigned, size_t>>& global_to_local_map,
        const Parameters &parameters)
    {
        bool print = parameters.Get<bool>("print");
        unsigned kbase = parameters.Get<unsigned>("kbase");

        size_t num_element = 0;
        for (unsigned i = 0; i < m; ++i)
        {
            num_element += graphs[i]->max_elements_;
        }

        if (num_element != kbase * max_elements_ or graphs.size() != m)
        {
            std::cerr << "Error: total number of elements or number of graphs does not match the initialized value." << std::endl;
            return;
        }

        num_element = max_elements_;

        // std::random_device rng;
        // std::mt19937 urng(rng());
        std::mt19937 urng(42);

        std::vector<bool> merged(num_element, 0);
        std::vector<bool> nhood_set(num_element, 0);
        std::vector<labeltype> final_nhood;

        unsigned short int nnbrs = 0, centroid_nnbrs = 0;
        labeltype cur_id = 0;
        for (std::pair<labeltype, unsigned> gid_cid : assignments)
        {
            labeltype global_label = gid_cid.first; // merge
            unsigned centroid_id = gid_cid.second;
            labeltype local_label = global_to_local_map[global_label][centroid_id];
            HierarchicalNSW<dist_t>* graph = graphs[centroid_id];
            tableint local_internal_id = graph->label_lookup_[local_label];

            if (cur_id < global_label)
            {
                cur_element_count ++;
                assert(cur_id == global_label - 1);
                std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
                nnbrs = (unsigned short int)(std::min)(final_nhood.size(), maxM0_);

                memset(data_level0_memory_ + cur_id * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
                // Initialisation of the data and label
                memcpy(getExternalLabeLp(cur_id), &cur_id, sizeof(labeltype)); // label = external id = partition id, writing to cur_c's label
                memcpy(getDataByInternalId(cur_id), data + cur_id * dim, data_size_); // internal id: row number of level0's index layout, writing cur_c's vector
                linklistsizeint* ll_cur = get_linklist0(cur_id);
                setListCount(ll_cur, nnbrs);
                tableint *linklist = (tableint*)(ll_cur + 1);
                for (unsigned i = 0; i < nnbrs; ++i)
                {
                    if (linklist[i])
                    {
                        throw std::runtime_error("Possible memory corruption: linklist[i] is not empty");
                    }
                    linklist[i] = final_nhood[i];
                }
                if (print && cur_id % 49999 == 1)
                {
                    std::cout << "\rmerged " << 100 * cur_id / num_element << " %..." << std::flush;
                }
                cur_id = global_label;
                nnbrs = 0;
                for (labeltype &p : final_nhood)
                    nhood_set[p] = 0;
                final_nhood.clear();
            }

            linklistsizeint* cur_element_data = graph->get_linklist0(local_internal_id);
            centroid_nnbrs = graph->getListCount(cur_element_data);

            if (centroid_nnbrs == 0)
            {
                std::cerr << "Warning: centroid " << centroid_id << " , internal_node " << local_internal_id << " has no neighbor!" << std::endl;
            }

            std::vector<labeltype> centroid_nhood(centroid_nnbrs);
            for (unsigned i = 0; i < centroid_nnbrs; ++i)
            {
                tableint neighbor_internalid = *((tableint*)(cur_element_data + 1) + i);
                labeltype neighbor_local_label = graph->getExternalLabel(neighbor_internalid);
                centroid_nhood[i] = neighbor_local_label;
            }

            for (unsigned i = 0; i < centroid_nnbrs; ++i)
            {
                if (nhood_set[idmaps[centroid_id][centroid_nhood[i]]] == 0)
                {
                    nhood_set[idmaps[centroid_id][centroid_nhood[i]]] = 1;
                    final_nhood.emplace_back(idmaps[centroid_id][centroid_nhood[i]]);
                }
            }
        }
        cur_element_count ++;
        assert(cur_id == num_element - 1);
        assert(cur_element_count == num_element);
        std::shuffle(final_nhood.begin(), final_nhood.end(), urng);
        nnbrs = (unsigned short int)(std::min)(final_nhood.size(), maxM0_);

        memset(data_level0_memory_ + cur_id * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);
        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_id), &cur_id, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_id), data + cur_id * dim, data_size_);
        linklistsizeint* ll_cur = get_linklist0(cur_id);
        setListCount(ll_cur, nnbrs);
        tableint *linklist = (tableint*)(ll_cur + 1);
        for (unsigned i = 0; i < nnbrs; ++i)
        {
            if (linklist[i])
            {
                throw std::runtime_error("Possible memory corruption: linklist[i] is not empty");
            }
            linklist[i] = final_nhood[i];
        }

        maxlevel_ = std::numeric_limits<int>::min();

        for (unsigned cid = 0; cid < m; ++cid)
        {
            HierarchicalNSW<dist_t>* graph = graphs[cid];
            if (graph->maxlevel_ > maxlevel_) {
                maxlevel_ = graph->maxlevel_;
                labeltype graph_ep = graph->getExternalLabel(graph->enterpoint_node_);
                enterpoint_node_ = idmaps[cid][graph_ep];
            }
        }

        maxlevel_ = 0;

        std::cout << std::endl;
    }

    void update_hits_counter(const std::vector<tableint>& selectedNeighbors,
                        const std::pair<size_t, size_t>& hit_range,
                        std::vector<std::atomic<uint8_t>>& hits,
                        std::atomic<size_t>& total_hits)
    {
        for (tableint selected_id : selectedNeighbors)
        {
            if (selected_id >= hit_range.first && selected_id < hit_range.second) {
                size_t idx = selected_id - hit_range.first;
                uint8_t expected = 0;
                if (hits[idx].compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                    total_hits.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    void NGM_merge_into_later(std::vector<HierarchicalNSW<dist_t>*> graphs, unsigned G1_id, unsigned G2_id, size_t ef_merge, const Parameters &parameters)
    {
        HierarchicalNSW<dist_t>* G1 = graphs[G1_id];
        HierarchicalNSW<dist_t>* G2 = graphs[G2_id];

        // TODO: no use parameters
        bool et = parameters.Get<bool>("early_terminate");
        float ratio = parameters.Get<float>("et_ratio");
        size_t target_hits = static_cast<size_t>(ratio * G2->cur_element_count);

        std::pair<size_t, size_t> hit_range = {globalid_offset[G2_id], globalid_offset[G2_id] + G2->cur_element_count};// [A,B)
        std::vector<std::atomic<uint8_t>> hits(G2->cur_element_count);
        for (auto& h : hits) h.store(0, std::memory_order_relaxed);
        std::atomic<size_t> total_hits{0};
        std::atomic<bool> should_terminate{false};


#pragma omp parallel for schedule(dynamic, 72)
        for (tableint internal_id = 0; internal_id < G1->cur_element_count; ++internal_id)
        {
            if (et && should_terminate.load(std::memory_order_acquire)) {
                continue; // TODO: no use code
            }

            float* data_point = (float*) G1->getDataByInternalId(internal_id);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, HierarchicalNSW<float>::CompareByFirst>
            temp_candidates = G2->Global_merge(data_point, ef_merge);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

            // pruning candidateset -> internal id of merged graph
            while (!temp_candidates.empty()) {
                auto candidate = temp_candidates.top();
                temp_candidates.pop();
                top_candidates.push({candidate.first, candidate.second + globalid_offset[G2_id]});
            }
            tableint merged_internal_id = internal_id + globalid_offset[G1_id];

            // top_candidates <-- original neighbors
            linklistsizeint* cur_element_data = get_linklist0(merged_internal_id);
            unsigned short int neighbor_count = getListCount(cur_element_data);
            for (unsigned i = 0; i < neighbor_count; ++i)
            {
                tableint neighbor_internalid = *((tableint*)(cur_element_data + 1) + i);
                top_candidates.push({fstdistfunc_(data_point, getDataByInternalId(neighbor_internalid), dist_func_param_), neighbor_internalid});
            }

            std::vector<tableint> selectedNeighbors = Pruning_InterInsert(data_point, merged_internal_id, top_candidates, 0); // updata merged_internal_id's neighborhood and adding reverse edges
            if (et)
            {
                for (tableint selected_id: selectedNeighbors)
                {
                    if (selected_id >= hit_range.first && selected_id < hit_range.second) {
                        size_t idx = selected_id - hit_range.first;
                        uint8_t expected = 0;
                        if (hits[idx].compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                            total_hits.fetch_add(1, std::memory_order_relaxed);

                            if (total_hits >= target_hits) {
                                should_terminate.store(true, std::memory_order_release);
                                break;
                            }
                        }
                    }
                }
            }

        }
    }

    std::vector<block_info> construct_blocks(HierarchicalNSW<dist_t>* G, const Parameters &parameters)
    {
        unsigned self_ef = parameters.Get<unsigned>("self_ef");

        // step1 get kNN
        std::vector<std::vector<tableint>> Nks;
        Nks.resize(G->cur_element_count);
#pragma omp parallel for schedule(dynamic, 72)
        for (tableint internal_id = 0; internal_id < G->cur_element_count; ++internal_id)
        {
            float* data_point = (float*) G->getDataByInternalId(internal_id);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, HierarchicalNSW<float>::CompareByFirst>
            temp_candidates = G->Self_search(data_point, self_ef, internal_id);

            while (temp_candidates.size() != 0)
            {
                Nks[internal_id].push_back(temp_candidates.top().second);
                temp_candidates.pop();
            }
        }

        // step2 construct reverse NN
        std::vector<reverseNN_info> Rks;
        Rks.resize(G->cur_element_count);
        std::vector<std::mutex> rks_mutexes(G->cur_element_count);  // lock for each Rks entry

        for (tableint internal_id = 0; internal_id < G->cur_element_count; ++internal_id) {
            Rks[internal_id] = reverseNN_info(internal_id);
        }

#pragma omp parallel for schedule(dynamic, 72)
        for (tableint internal_id = 0; internal_id < G->cur_element_count; ++internal_id) {
            for (tableint neighbor_id : Nks[internal_id]) {
                std::lock_guard<std::mutex> lock(rks_mutexes[neighbor_id]);
                Rks[neighbor_id].rNNs.push_back(internal_id);
                Rks[neighbor_id].length++;
            }
        }

        // step3 sort Rks by size
        tbb::parallel_sort(Rks.begin(), Rks.end(),
                                 [](const reverseNN_info& a, const reverseNN_info& b) {
                                     return a.length > b.length;
                                 });

        // step4 construct blocks
        std::vector<unsigned> considered;
        considered.resize(G->cur_element_count, 0);
        std::vector<block_info> blocks;
        blocks.reserve(G->cur_element_count);
        for (tableint internal_id = 0; internal_id < G->cur_element_count; ++internal_id) {
            tableint bid = Rks[internal_id].id;
            if (considered[bid] == 1) continue;

            considered[bid] = 1;
            std::vector<tableint> bmembers;
            bmembers.reserve(Rks[internal_id].rNNs.size());
            for (tableint c : Rks[internal_id].rNNs) {
                if (considered[c] == 0) {
                    considered[c] = 1;
                    bmembers.push_back(c);
                }
            }

            blocks.push_back(block_info(bid, bmembers));
        }

        return blocks;
    }

    void RGTM_merge_into_later(std::vector<HierarchicalNSW<dist_t>*> graphs, unsigned G1_id, unsigned G2_id, const Parameters &parameters)
    {
        HierarchicalNSW<dist_t>* G1 = graphs[G1_id];
        HierarchicalNSW<dist_t>* G2 = graphs[G2_id];
        bool print = parameters.Get<bool>("print");

        // parameters for global merge and local sliding
        unsigned global_ef = ef_construction_;
        unsigned local_ef = parameters.Get<unsigned>("local_ef");

        // TODO: no use parameters
        bool et = parameters.Get<bool>("early_terminate");
        float ratio = parameters.Get<float>("et_ratio");
        size_t target_hits = static_cast<size_t>(ratio * G2->cur_element_count);
        std::pair<size_t, size_t> hit_range = {globalid_offset[G2_id], globalid_offset[G2_id] + G2->cur_element_count};// [A,B)
        std::vector<std::atomic<uint8_t>> hits(G2->cur_element_count);
        for (auto& h : hits) h.store(0, std::memory_order_relaxed);
        std::atomic<size_t> total_hits{0};
        std::atomic<bool> should_terminate{false};

        auto blocks = construct_blocks(G1, parameters);

        if (print) {
            size_t G_cnt = 0;
            size_t L_cnt = 0;

            G_cnt = blocks.size();
            for (auto block : blocks) {
                L_cnt += block.bmembers.size();
            }
            std::cout << "L : G = " << L_cnt << " : " << G_cnt << " = " << (float)L_cnt/G_cnt << std::endl;
        }

        // start merging
#pragma omp parallel for schedule(dynamic, 72)
        for (size_t i = 0; i < blocks.size(); ++i) // Global Merge
        {
            if (et && should_terminate.load(std::memory_order_acquire)) {
                continue; // TODO: no use code
            }

            tableint global_search_id = blocks[i].bid;
            std::vector<tableint> local_search_members = blocks[i].bmembers;
            float* global_merge_data = (float*) G1->getDataByInternalId(global_search_id);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, HierarchicalNSW<float>::CompareByFirst>
            temp_candidates = G2->Global_merge(global_merge_data, global_ef);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

            // pruning candidateset -> internal id of merged graph
            std::vector<tableint> starting_ids;
            while (!temp_candidates.empty()) {
                auto candidate = temp_candidates.top();
                temp_candidates.pop();
                starting_ids.push_back(candidate.second);
                top_candidates.push({candidate.first, candidate.second + globalid_offset[G2_id]});
            }
            tableint merged_internal_id = global_search_id + globalid_offset[G1_id];

            // top_candidates <-- original neighbors
            linklistsizeint* cur_element_data = get_linklist0(merged_internal_id);
            unsigned short int neighbor_count = getListCount(cur_element_data);
            for (unsigned k = 0; k < neighbor_count; ++k)
            {
                tableint neighbor_internalid = *((tableint*)(cur_element_data + 1) + k);
                top_candidates.push({fstdistfunc_(global_merge_data, getDataByInternalId(neighbor_internalid), dist_func_param_), neighbor_internalid});
            }

            std::vector<tableint> selectedNeighbors = Pruning_InterInsert(global_merge_data, merged_internal_id, top_candidates, 0); // updata merged_internal_id's neighborhood and adding reverse edges

            if (et)
            {
                update_hits_counter(selectedNeighbors, hit_range, hits, total_hits);
            }

            for (size_t j = 0; j < local_search_members.size(); ++j) // Local Sliding
            {
                tableint local_search_id = local_search_members[j];
                float* local_merge_data = (float*) G1->getDataByInternalId(local_search_id);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, HierarchicalNSW<float>::CompareByFirst>
                local_temp_candidates = G2->Local_merge(local_merge_data, local_ef, starting_ids);
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> local_top_candidates;

                // pruning candidateset --> internal id of merged G
                while (!local_temp_candidates.empty()) {
                    auto candidate = local_temp_candidates.top();
                    local_temp_candidates.pop();
                    local_top_candidates.push({candidate.first, candidate.second + globalid_offset[G2_id]});
                }
                tableint local_merged_internal_id = local_search_id + globalid_offset[G1_id];

                // top_candidates <-- original neighbors
                linklistsizeint* local_cur_element_data = get_linklist0(local_merged_internal_id);
                unsigned short int local_neighbor_count = getListCount(local_cur_element_data);
                for (unsigned k = 0; k < local_neighbor_count; ++k)
                {
                    tableint neighbor_internalid = *((tableint*)(local_cur_element_data + 1) + k);
                    local_top_candidates.push({fstdistfunc_(local_merge_data, getDataByInternalId(neighbor_internalid), dist_func_param_), neighbor_internalid});
                }

                std::vector<tableint> local_selectedNeighbors = Pruning_InterInsert(local_merge_data, local_merged_internal_id, local_top_candidates, 0); // updata local_merged_internal_id's neighborhood and adding reverse edges

                if (et)
                {
                    update_hits_counter(selectedNeighbors, hit_range, hits, total_hits);
                }
            }

            if (total_hits >= target_hits) {
                should_terminate.store(true, std::memory_order_release);
            }
        }
        if (et) {
            std::cout << "ET early terminate at " << total_hits.load(std::memory_order_relaxed) << " hits." << std::endl;
        }
    }

    void fxy_merge(unsigned m, std::vector<HierarchicalNSW<dist_t>*> graphs, const Parameters &parameters)
    {
        std::string method = parameters.Get<std::string>("method");

        size_t num_element = 0;
        for (unsigned i = 0; i < m; ++i)
        {
            num_element += graphs[i]->max_elements_;
        }

        if (num_element != max_elements_ or graphs.size() != m)
        {
            std::cerr << "Error: total number of elements or number of graphs does not match the initialized value." << std::endl;
            return;
        }

        initialize_mergeid_lookup(graphs);
        init_merge_graph_level0(graphs);

        // merge order selection
        std::vector<std::pair<unsigned, unsigned>> merge_order;
        pairwise_merge_order(m, merge_order);

        // start merging
        if (method == "NGM")
        {
            for (auto&& p : merge_order) { // pairwise merge
                if (parameters.Get<bool>("reverse"))
                {
                    NGM_merge_into_later(graphs, p.second, p.first, ef_construction_, parameters); // second merge into first
                }
                else
                {
                    NGM_merge_into_later(graphs, p.first, p.second, ef_construction_, parameters); // first merge into second
                }
            }
        }
        else if (method == "RGTM")
        {
            for (auto&& p : merge_order) { // pairwise merge
                RGTM_merge_into_later(graphs, p.first, p.second, parameters); // first merge into second
                RGTM_merge_into_later(graphs, p.second, p.first, parameters); // second merge into first
            }
        }
        else
        {
            std::cerr << "Error: unknown method " << method << std::endl;
        }
    }
};
}  // namespace hnswlib

