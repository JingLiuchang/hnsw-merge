#!/bin/bash
#scp -r /mnt/ssd/merge_bench/anton1m/random/bi-index-data/anton1m_random_base.fvecs bld@10.4.15.162:/home/bld/nsg-lab/search-lab/PyRemote/dataprocess/merge_bench/
#scp -r /mnt/ssd/merge_bench/anton1m/random/bi-index-data/anton1m_randomP1_base.fvecs bld@10.4.15.162:/home/bld/nsg-lab/search-lab/PyRemote/dataprocess/merge_bench/
#scp -r /mnt/ssd/merge_bench/imagenet1m/random/bi-index-data/imagenet1m_random_base.fvecs bld@10.4.15.162:/home/bld/nsg-lab/search-lab/PyRemote/dataprocess/merge_bench/
#scp -r /mnt/ssd/merge_bench/imagenet1m/random/bi-index-data/imagenet1m_randomP1_base.fvecs bld@10.4.15.162:/home/bld/nsg-lab/search-lab/PyRemote/dataprocess/merge_bench/

source params.sh
for db in "${datasets[@]}"; do
  scp /mnt/ssd/merge_bench/${db}/random/bi-index-data/${db}_random_base.fvecs /mnt/ssd/merge_bench/${db}/${db}_query.fvecs bld@10.4.15.162:/home/bld/nsg-lab/search-lab/PyRemote/dataprocess/merge_bench/
done