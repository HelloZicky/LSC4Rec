#!/bin/bash
#sleep 27000
date
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=("beauty" "cds" "electronic")
#dataset_list=("beauty" "cds" "electronic" "yelp")
#dataset_list=("beauty" "sports" "toys" "yelp")
#dataset_list=("beauty" "sports" "yelp")
# dataset_list=("toys")
# dataset_list=("yelp")
#dataset_list=("sports" "yelp")
# dataset_list=("beauty")
#dataset_list=("beauty" "sports")
dataset_list=("beauty")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(0 1 2 3)
#cuda_num_list=(0 1 2)
#cuda_num_list=(2 3)
#cuda_num_list=(2 3 4)
#cuda_num_list=(4 5 6)
cuda_num_list=(3)
echo ${line_num_list}
seed=2022
length=${#dataset_list[@]}
#for dataset line_num in zip()
for ((i=0; i<${length}; i++));
#for i in {0..${length}-1};
do
{
    dataset=${dataset_list[i]}
#    line_num=${line_num_list[i]}
    cuda_num=${cuda_num_list[i]}
#    for model in din gru4rec sasrec
#    for model in t5-base t5-small
   for model in t5-small
    # for model in t5-base
    do
    {
#        for split in train evaluate
        for split in train
        do
          {
#              for type in all traditional sequential rating
#              for type in all traditional
#              for type in all
              # for type in traditional
             for type in sequential
#              for type in rating
              do
                {
                  set |grep RANDOM
                  # 0~65536
                  port=$[${RANDOM}%10000+30000]
                  echo $port
                  if [ ${dataset} = "yelp" ];then
                    learning_rate=1e-3
                    warmup_ratio=0.02
                  else
                    learning_rate=1e-3
                    warmup_ratio=0.05
                  fi

#                  file="pretrain_${type}_${split}"
                  file="pretrain_llm_train"

                  bash scripts/${file}.sh ${dataset} ${model} ${cuda_num} ${port} ${warmup_ratio} ${learning_rate} ${seed} ${type}

#                  cd ../
                } &
              done
          } &
        done
    } &
    done
} &
done
wait # 等待所有任务结束
date
