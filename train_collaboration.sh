#!/bin/bash
#sleep 30000
date
dataset_list=("beauty")
# dataset_list=("sports")
# dataset_list=("toys")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(0 1 2 3)
#cuda_num_list=(0 1 2)
cuda_num_list=(3)
# recommendation_model_list=("din" "gru4rec" "sasrec")
recommendation_model_list=("sasrec")
echo ${line_num_list}
seed=2022
length=${#dataset_list[@]}
recommendation_model_length=${#recommendation_model_list[@]}
#for dataset line_num in zip()
for ((i=0; i<${length}; i++));
#for i in {0..${length}-1};
do
{
    dataset=${dataset_list[i]}
    for model in t5-small
#    for model in t5-base
    do
    {
#        for split in train evaluate
        for split in train
        do
          {
#              for type in all traditional sequential rating
#              for type in all traditional
#              for type in all
#              for type in traditional
              for type in sequential
#              for type in rating
              do
                {
                  for ((j=0; j<${recommendation_model_length}; j++));
#                  for recommendation_model in din gru4rec sasrec
#                  for recommendation_model in sasrec
#                  for recommendation_model in gru4rec
#                  for recommendation_model in din
                  do
                    {
#                        for type_small in base duet
                        recommendation_model=${recommendation_model_list[j]}
#                        ${cuda_num_list[i]}
                        cuda_num=${cuda_num_list[j]}
                        for type_small in base
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
            #                  file="pretrain_llm_train"
#                              file="pretrain_collaboration_train"
                              file="pretrain_collaboration_aug_train"
                              bash scripts/${file}.sh ${dataset} ${model} ${cuda_num} ${port} ${warmup_ratio} ${learning_rate} ${seed} ${type} ${recommendation_model} ${type_small} 2
                          } &
                        done
                    } &
                  done
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
