# Parallelism-Exp 算法开发准备工作

## 当前状态
✅ 已创建分支 `parallelism-exp`
✅ 当前修改已带入新分支

## 新目标

开发一个新的merge算法，neighbor sliding merge，简称NSM，作为parallelism-exp这个实验中新实现的一个用于和已有的RNSM算法进行比较的的核心算法。

## 任务说明

1. **RNSM说明**：
    # 理解 RNSM 算法
    
    # 背景问题：图合并的挑战
    
    RNSM 解决的是**两个近似近邻图（Proximity Graph）的合并**问题：给定源图 $G_1$ 和目标图 $G_2$，需要将 $V_1$ 中的每个节点都在 $G_2$ 中找到其近邻，从而构建统一的合并图 $G$。
    
    核心难点在于：
    - **暴力搜索**：对每个节点都从默认入口在 $G_2$ 上做全量搜索，代价极高
      - **滑动搜索**：若节点 $y$ 与节点 $x$ 相近，则可以用 $x$ 在 $G_2$ 上的搜索结果作为 $y$ 的入口点（Local Sliding），节省代价
      - **但是**：滑动产生了**依赖链**（$y$ 必须等 $x$ 先搜索完），无法并行化
    
    ---
    
    # RNSM 的核心思路
    
    > **在合并执行之前，全局地、一次性地完成 pivot 选择，从而彻底解耦依赖，实现完全并行。**
    
    利用 **RNN（反向 k 近邻）图**的性质：若 $x \in N_k(y)$（$x$ 是 $y$ 的近邻），则反过来 $x$ 天然是 $y$ 的一个优质 pivot（因为它们空间上相近）。
    
    拥有大量 RNN 的节点 = **自然的 Hub 节点** = 可以高效地服务于众多邻居的 pivot。
    
    ---
    
    # 三阶段详解
    
    ## 阶段一：预计算（Pre-computation）
    
    | 步骤 | 目的 |
    |------|------|
    | **邻域扩展** | $G_1$ 是稀疏的近似图，通过图遍历将每个节点的邻域扩展到 $k^+$，使候选 pivot 更准确 |
    | **反向索引** | 构建 $R_k(x) = \{y \mid x \in N_k(y)\}$，即"有多少节点把 $x$ 当近邻" |
    | **排序** | $\|R_k(x)\|$ 越大，$x$ 越适合当 pivot（可覆盖更多节点）|
    
    ## 阶段二：贪心 Pivot 选择（Pivot Selection）
    
    **直觉**：按 RNN 度数从大到小贪心选取 pivot，每个 pivot 负责"带着"自己的反向邻居（followers）。
    - 每个节点**只被覆盖一次**
      - pivot 与 followers 之间空间上**天然相近**（由 RNN 性质保证）
    
    ## 阶段三：并行合并（Parallel Merging）
    
    - **Pivot** 做完整的 Naive Search（无依赖，可完全并行）
      - **Followers** 以 pivot 的搜索结果为入口做 Local Sliding（代价低）
      - 由于所有 pivot 互相独立，整个过程**无数据依赖、完全并行**
    
    ---
    
    # 完整伪代码
   \begin{algorithm}[!t]
   \caption{Reverse Neighbor Sliding Merge (RNSM)}
   \label{algo:bi-merge}
   \begin{small}
   \KwIn{Graph indexes $G_1=\left(V_1,E_1\right)$ and $G_2=\left(V_2,E_2\right)$, $k$}
   \KwOut{Merged graph index $G=\left(V,E\right)$}
   $V \gets V_1 \cup V_2$\;
   $P \gets \emptyset$\tcp*{Pivots Set}
   $N_p \gets \emptyset$\tcp*{Pivots Neighbor Set}
   Expand $N_k(x)$ with $k^+$ result\tcp*{Neighbor Expand}
   Construct $R_k(x)$ from $N_k(x)$\tcp*{Reverse KNN}
   Sort $x\in V_1$ in descending order of $|R_k(x)|$\;
   \For{$x\in V_1$}{
       \If{$x\notin P$ and $x\notin N_p(p)$ where $p\in P$}{
           $P \gets P \cup x$\;
           $R'_k(x) \gets R_k(x)-V_{\text{merged}}$\;
       }
   }

    \For{$x\in P$ \text{\textbf{parallel}}}{
        $S_N \gets \mathrm{Search\mbox{-}Graph}(G_2,x)$\tcp*{Naive Search}
        $C \gets S_N \cup \operatorname{Neighbor}(x,E_1)$\;
        $G \gets \mathrm{UpdateNeighborhood}(x,C)$\;
        \For{$y\in R'_k(x)$}{
            $S_L \gets \mathrm{Search\mbox{-}Graph}(G_2,S_N,y)$\tcp*{Sliding}
            $C \gets S_L \cup \operatorname{Neighbor}(y,E_1)$\;
            $G \gets \mathrm{UpdateNeighborhood}(y,C)$\;
        }
    }
    \Return{$G$}
    \end{small}
    \end{algorithm}
2. **NSM说明**：
    # 理解 NSM 算法
   优化之前的算法可以叫做neighbor sliding merge。
   其核心思想是随机选一个G1中的pivot a，在G2中对其做naive search得到结果集合Ra，然后对pivot a做neighbor spanding，从spanding结果中找出“未被cover过的top-k最近邻之一”作为follower 1。
   对follower 1，他得到pivot a的搜索结果Ra做Local Sliding得到结果R1。然后对follower 1做neighbor spanding，同样从spanding结果中找出“未被cover过的top-k最近邻之一”作为下一个follower：follower 2。
   对follower 2，他得到follower 1的搜索结果R1做Local Sliding得到结果R2，依次类推，直到对于某个follower i做neighbor spanding后找不到“未被cover过的top-k最近邻之一”，这时就又随机挑选一个未被cover过的pivot重新开始naive search。
   **由于这个算法形成链式的依赖关系，因此期待上我们认为它由于依赖关系导致的需要加锁，所以并行性能不如RNSM。这也是这个parallelism-exp分支要验证的关键点**

    # 完整伪代码
   \begin{algorithm}[!t]
   \caption{Neighbor Sliding Merge (NSM)}
   \label{algo:nsm}
   \begin{small}
   \KwIn{Graph indexes $G_1=\left(V_1,E_1\right)$ and $G_2=\left(V_2,E_2\right)$, $k$}
   \KwOut{Merged Graph $G=(V,E)$}
   $V \gets V_1 \cup V_2$; $E \gets E_1 \cup E_2$\;
   $Covered \gets \emptyset$\;
   \While{$Covered \neq V_1$}{
   $a \gets \mathrm{RandomPick}(V_1 \setminus Covered)$\;
   $Covered \gets Covered \cup \{a\}$\;

   $Res_{prev} \gets \mathrm{Search\mbox{-}Graph}(G_2, a)$\tcp*{Naive Search}
   $C \gets Res_{prev} \cup \operatorname{Neighbor}(a,E_1)$\;
   $G \gets \mathrm{UpdateNeighborhood}(a,C)$\;

   $current \gets a$\;

   \While{\textbf{True}}{
   $Expanded \gets \mathrm{ExpandNeighbors}(G_1, current, k^+)$\tcp*{Neighbor Expand}
   $TopK \gets \mathrm{GetTopK}(Expanded, \text{reference}=current, k=k)$\;
   $next \gets \mathrm{NearestUncovered}(TopK, Covered)$\;

   \If{$next = \emptyset$}{
   \textbf{break}\;
   }

   $Covered \gets Covered \cup \{next\}$\;
   $Res_{next} \gets \mathrm{Search\mbox{-}Graph}(G_2, Res_{prev}, next)$\tcp*{Sliding}
   $C \gets Res_{next} \cup \operatorname{Neighbor}(next,E_1)$\;
   $G \gets \mathrm{UpdateNeighborhood}(next,C)$\;

   $Res_{prev} \gets Res_{next}$\;
   $current \gets next$\;
   }
   }
   \Return{$G$}
   \end{small}
   \end{algorithm}
3. **任务目标**：
    - [ ] 理解 NSM 算法
    - [ ] 在example/cpp目录下实现test_NSM_merge.cpp作为NSM测试入口
    - [ ] 在mergealg.h中同样使用mgraph_merge函数作为merge算法选择的统一入口，并在其中添加NSM算法的选择分支
    - [ ] 在mergealg.cpp中实现NSM算法的具体逻辑，命名为NSM_merge_into_later