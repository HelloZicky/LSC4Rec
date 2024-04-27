#!/bin/bash
date
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=("beauty" "cds" "electronic")
#dataset_list=("beauty" "cds" "electronic" "yelp")
#dataset_list=("beauty" "sports" "toys" "yelp")
#dataset_list=("beauty" "sports" "yelp")
#dataset_list=("sports" "yelp")
dataset_list=("beauty" "sports")
#dataset_list=("sports")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(0 1 2 3)
#cuda_num_list=(0 1 2)
#cuda_num_list=(2 3)
#cuda_num_list=(2 3 4)
cuda_num_list=(5 6)
#cuda_num_list=(2)
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
    do
    {
#        for split in train evaluate
#        for split in train
#        do
#          {
#              for type in all traditional sequential rating
#              for type in all traditional
#              for type in all
#              for type in traditional
#              for type in sequential
#              do
#                {
#                  bash small_test.sh ${dataset} ${model}
#
##                  cd ../
#                } &
#              done
#              for framework in small_recommendation_model
              for framework in large_language_model
              do
                {
#                  for losses in sequential traditional all
#                  for losses in sequential traditional rating
#                  for losses in rating
                  for losses in sequential traditional
                  do
                    {
#                        for epoch in 1 2 3 4 5 6 7 8 9 10
#                          do
#                            {
#                              bash func_large_test.sh ${dataset} ${framework} ${model} ${cuda_num} ${losses} ${epoch}
                              bash func_large_all_test.sh ${dataset} ${framework} ${model} ${cuda_num} ${losses}
#                            } &
#                          done
                    } &
                  done

#                  cd ../
                } &
              done
#          } &
#        done
    } &
    done
} &
done
wait # 等待所有任务结束
date
