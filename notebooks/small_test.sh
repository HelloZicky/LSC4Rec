#!/bin/bash
date
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=("beauty" "cds" "electronic")
#dataset_list=("beauty" "cds" "electronic" "yelp")
#dataset_list=("beauty" "sports" "toys" "yelp")
# dataset_list=("beauty" "sports" "yelp")
#dataset_list=("sports" "yelp")
# dataset_list=("sports" "yelp")
dataset_list=("beauty")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(0 1 2 3)
#cuda_num_list=(0 1 2)
#cuda_num_list=(2 3)
# cuda_num_list=(2 3 4)
cuda_num_list=(2)
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
    for model in sasrec
    do
    {
              for framework in small_recommendation_model
              do
                {
                  for type in base
                  do
                    {
                        framework=${framework}_${type}
                        bash func_small_test.sh ${dataset} ${framework} ${model} ${cuda_num} ${type}
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
# wait # 等待所有任务结束
# date
#            } &
#                           done
#                     } &
#                   done
# #                  cd ../
#                 } &
#               done
# #          } &
# #        done
#     } &
#     done
# } &
# done
# wait # 等待所有任务结束
# date
