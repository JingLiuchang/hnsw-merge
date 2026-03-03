void IndexMerge::parallel_IGTM(IndexNSG* indexA, IndexNSG* indexB, IndexNSG* indexM, Parameters& parameters) {
std::cout << "Merging two NSG indices with IGTM..." << std::endl;
unsigned range = parameters.Get<unsigned>("R");
unsigned maxc = parameters.Get<unsigned>("C");

unsigned m = parameters.Get<unsigned>("m");
unsigned M = parameters.Get<unsigned>("M");
unsigned jump_ef = parameters.Get<unsigned>("jump_ef");
unsigned local_ef = parameters.Get<unsigned>("local_ef");
unsigned next_step_k = parameters.Get<unsigned>("next_step_k");
unsigned next_step_ef = parameters.Get<unsigned>("next_step_ef");

efanna2e::Parameters global_parameters, local_parameters, next_step_parameters;
global_parameters.Set<unsigned>("L",jump_ef);
local_parameters.Set<unsigned>("L",local_ef);
next_step_parameters.Set<unsigned>("L",next_step_ef);

size_t base = indexA->nd_; // indexB的id偏移量

SimpleNeighbor* cut_graph_ = new SimpleNeighbor[indexM->nd_ * (size_t)range];

auto s = std::chrono::high_resolution_clock::now();

// 第一阶段：处理indexA中的点 - 并行化修改
{
boost::dynamic_bitset<> not_doneA(indexA->nd_);  //First phase completed in 480716 ms.
not_doneA.set();
std::unordered_set<unsigned> initial_ids;
initial_ids.reserve(indexA->nd_);
for (unsigned i = 0; i < indexA->nd_; i++) {
initial_ids.insert(i);
}

    // 用于线程安全操作的互斥锁
    std::mutex not_done_mutex;

    #pragma omp parallel
    {
      std::vector<Neighbor> pool;
      std::vector<Neighbor> tmp;
      boost::dynamic_bitset<> flags{indexB->nd_, 0};

      while(true) {
        // 安全地获取下一个待处理ID
        unsigned current_id;
        bool has_work = false;

        {
          std::lock_guard<std::mutex> lock(not_done_mutex);
          if (!initial_ids.empty()) {
            current_id = *initial_ids.begin();
            initial_ids.erase(current_id);
            not_doneA[current_id] = false;
            has_work = true;
          }
        }

        if (!has_work) break; // 没有更多工作，退出循环

        pool.clear();
        tmp.clear();
        flags.reset();

        indexB->get_neighbors(indexA->data_ + indexA->dimension_ * current_id, global_parameters, flags, tmp, pool);
        for(auto& t : tmp) {
          t.id += base;
        }
        std::vector<Neighbor> starting_points = tmp;

        unsigned cnt = 0;

        while (true) {
          std::vector<Neighbor> Cb;
          std::vector<Neighbor> C;
          boost::dynamic_bitset<> local_flags{indexB->nd_, 0};

          // 第一次处理当前点，直接使用外层的search结果
          if (cnt == 0) {
            Cb = starting_points;
            Cb.resize(m);
            C = Cb;
          } else {
            Cb = indexB->local_get_neighbors(indexA->data_ + indexA->dimension_ * current_id, local_parameters, local_flags, starting_points).second;
            std::sort(Cb.begin(), Cb.end());
            Cb.resize(m);

            for (auto& t : Cb) {
              t.id += base;
            }

            C = Cb;
          }

          cnt += 1;

          // Pb + N(va)
          for(unsigned z = 0; z < indexA->final_graph_[current_id].size(); z++) {
            unsigned id = indexA->final_graph_[current_id][z];

            float dist = indexA->distance_->compare(
              indexA->data_ + indexA->dimension_ * current_id,
              indexA->data_ + indexA->dimension_ * id,
              indexA->dimension_);
            C.push_back(Neighbor(id, dist, true));
          }

          // 剪枝过程
          std::vector<Neighbor> result;
          unsigned start = 0;

          std::sort(C.begin(), C.end());
          if(C[start].id == current_id) start++;
          if(start < C.size()) result.push_back(C[start]);

          while(result.size() < range && (++start) < C.size() && start < maxc) {
            auto& p = C[start];
            bool occlude = false;

            for(unsigned t = 0; t < result.size(); t++) {
              if(p.id == result[t].id) {
                occlude = true;
                break;
              }
              float djk = indexM->distance_->compare(
                indexM->data_ + indexM->dimension_ * result[t].id,
                indexM->data_ + indexM->dimension_ * p.id,
                indexM->dimension_);
              if(djk < p.distance) {
                occlude = true;
                break;
              }
            }

            if(!occlude) result.push_back(p);
          }

          // 写入cut_graph_
          SimpleNeighbor* des_pool = cut_graph_ + (size_t)current_id * (size_t)range;
          for(size_t t = 0; t < result.size(); t++) {
            des_pool[t].id = result[t].id;
            des_pool[t].distance = result[t].distance;
          }
          if(result.size() < range) {
            des_pool[result.size()].distance = -1;
          }

          // starting points for next vertex
          starting_points = Cb;
          starting_points.resize(M);
          for (auto& t : starting_points) {
            t.id -= base;
          }

          // local search indexA find next vertex
          std::vector<Neighbor> current_node;
          current_node.push_back(Neighbor(current_id, 0.0f, true));
          local_flags.reset();
          auto Ca = indexA->local_get_neighbors(indexA->data_ + indexA->dimension_ * current_id, next_step_parameters, local_flags, current_node).first;

          Ca.resize(next_step_k);

          bool found_next = false;
          unsigned next_id = 0;

          // 找到下一个可能的节点
          for (auto& next_cur : Ca) {
            bool is_not_done;
            {
              std::lock_guard<std::mutex> lock(not_done_mutex);
              is_not_done = not_doneA[next_cur.id];
            }

            if (is_not_done) {
              next_id = next_cur.id;
              found_next = true;

              // 标记为已处理
              {
                std::lock_guard<std::mutex> lock(not_done_mutex);
                if (not_doneA[next_id]) {  // 再次检查，避免竞态条件
                  not_doneA[next_id] = false;
                  // 从initial_ids中移除（如果存在）
                  initial_ids.erase(next_id);
                } else {
                  found_next = false;  // 如果已经被其他线程标记，则不算找到
                }
              }

              if (found_next) {
                current_id = next_id;
                break;
              }
            }
          }

          if (!found_next) {
            break; // 没有找到下一个点，处理下一个起始点
          }
        }
      }
    }
}

auto e = std::chrono::high_resolution_clock::now();
std::cout << "First phase completed in "
<< std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
<< " ms." << std::endl;

s = std::chrono::high_resolution_clock::now();

for (unsigned current_id = base; current_id < indexM->nd_; current_id ++) {
SimpleNeighbor* des_pool = cut_graph_ + (size_t)(current_id) * (size_t)range;
auto result = indexB->final_graph_[current_id-base];
for(size_t t = 0; t < result.size(); t++) {
des_pool[t].id = result[t] + base;
des_pool[t].distance = indexB->distance_->compare(
indexB->data_ + indexB->dimension_ * (current_id-base),
indexB->data_ + indexB->dimension_ * result[t],
indexB->dimension_);
}
if(result.size() < range) {
des_pool[result.size()].distance = -1;
}
}



// 后续代码保持不变
s = std::chrono::high_resolution_clock::now();
std::vector<std::mutex> locks(indexM->nd_);
#pragma omp parallel for schedule(dynamic, 72)
for (unsigned n = 0; n < indexM->nd_; ++n) {
indexM->InterInsert(n, range, locks, cut_graph_);
}
e = std::chrono::high_resolution_clock::now();
std::cout << "Reverse edge insertion completed in "
<< std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
<< " ms." << std::endl;

// 构建indexM final_graph_
indexM->final_graph_.resize(indexM->nd_);
for (size_t i = 0; i < indexM->nd_; i++) {
SimpleNeighbor *pool = cut_graph_ + i * (size_t)range;
unsigned pool_size = 0;
for (unsigned j = 0; j < range; j++) {
if (pool[j].distance == -1) break;
pool_size = j;
}
pool_size++;
indexM->final_graph_[i].resize(pool_size);
for (unsigned j = 0; j < pool_size; j++) {
indexM->final_graph_[i][j] = pool[j].id;
}
}

s = std::chrono::high_resolution_clock::now();
// connectivity
indexM->tree_grow(parameters);
e = std::chrono::high_resolution_clock::now();
std::cout << "Connectivity tree construction completed in "
<< std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count()
<< " ms." << std::endl;

// 输出统计信息
unsigned max = 0, min = 1e6, avg = 0;
for (size_t i = 0; i < indexM->nd_; i++) {
auto size = indexM->final_graph_[i].size();
max = max < size ? size : max;
min = min > size ? size : min;
avg += size;
}
avg /= 1.0 * indexM->nd_;
printf("Degree Statistics: Max = %d, Min = %d, Avg = %d\n", max, min, avg);

indexM->has_built = true;
indexM->ep_ = indexA->ep_;
delete[] cut_graph_;

return ;
}