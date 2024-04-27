#!/bin/bash
date
dataset_list=("beauty")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
cuda_num_list=(2)
echo ${line_num_list}
seed=2022
length=${#dataset_list[@]}
for ((i=0; i<${length}; i++));
do
{
    dataset=${dataset_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in t5-small
    do
    {
        for split in train
        do
          {
                  for recommendation_model in  sasrec
                  do
                    {
                      for framework in small_recommendation_model
                      do
                        {
                           for type_small in base
                            do
                              {
                                  framework=${framework}_${type_small}
                                  set |grep RANDOM
                                  # 0~65536
                                  port=$[${RANDOM}%10000+20000]
                                  echo $port
                                  if [ ${dataset} = "yelp" ];then
                                    learning_rate=1e-3
                                    warmup_ratio=0.02
                                  else
                                    learning_rate=1e-3
                                    warmup_ratio=0.05
                                  fi
                                  # for type in traditional
                                 for type in sequential
                                  do
                                    {
          #                            file="pretrain_${type}_${split}"
                                      file="recommendation_train_retrain"
                                      bash scripts/${file}.sh ${dataset} ${model} ${cuda_num} ${port} ${warmup_ratio} ${learning_rate} ${seed} ${recommendation_model} ${framework} ${type_small} ${type}
                                    } &
                                  done
                              } &
                            done
                        } &
                        done
                    } &
                    done
#                } &
#              done
          } &
        done
    } &
    done
} &
done
wait # 等待所有任务结束
date

# bash scripts/pretrain.sh beauty ddd 1 11111 0 0.001 2022 sasrec small_recommendation_model
# bash scripts/pretrain_all_train.sh beauty t5-small 1 11113 0.05 1e-3 2022
# bash scripts/pretrain_sequential_train.sh beauty t5-small 1 11111 0.05 1e-3 2022
# bash scripts/recommendation_train.sh beauty ddd 7 11115 0 0.001 2022 sasrec small_recommendation_model
# bash scripts/recommendation_train.sh beauty ddd 7 11116 0 0.001 2022 din small_recommendation_model
# bash scripts/recommendation_train.sh beauty ddd 7 11117 0 0.001 2022 gru4rec small_recommendation_model