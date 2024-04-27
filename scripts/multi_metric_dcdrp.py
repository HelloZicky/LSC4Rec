import math
import os
import numpy
import numpy as np
from collections import defaultdict
import csv

ROOT_FOLDER = "../checkpoint/NIPS2023"
# DATASET_LIST = ["amazon_cds", "amazon_tv", "amazon_electronic", "amazon_beauty", "amazon_clothing", "amazon_books", "amazon_sports", "movielens", "movielens_100k", ]
# DATASET_LIST = ["amazon_cds", "amazon_electronic"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "douban_book", "douban_music"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "douban_book", "douban_music", "movielens_100k", "movielens_1m", "movielens_10m"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic",
#                 "douban_book", "douban_music",
#                 "movielens_100k", "movielens_1m"]
DATASET_LIST = ["amazon_cds"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic", "movielens_100k", "movielens_1m"]
# DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic"]
MODEL_LIST = ["din", "sasrec", "gru4rec"]

# TYPE_LIST = ["base", "base_finetune", "meta", "meta_gru", "meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_if", "meta_ood", "meta_ood2", "meta_ood_gru", "meta_ood_gru2"]
# TYPE_LIST = ["base", "base_finetune", "meta", "meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood", "meta_ood_uncertainty"]
# TYPE_LIST = ["base", "base_finetune", "meta", "meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty5_onlyu", "meta_ood_uncertainty5"]
# TYPE_LIST = ["base", "meta", "meta_grad", "meta_param"]
# TYPE_LIST = ["base", "meta", "meta_grad", "meta_param", "meta_grad_group_5", "meta_grad_group_10", "meta_grad_group_50"]
# TYPE_LIST = ["base", "2_prototype_long_tail", "meta", "meta_grad", "meta_grad_group_5", "meta_grad_group_10", "meta_grad_group_50", "meta_param", "meta_param_group_5", "meta_param_group_10", "meta_param_group_50"]
# TYPE_LIST = ["base", "2_prototype_long_tail", "meta", "meta_grad", "meta_grad_group_5", "meta_grad_group_10", "meta_grad_group_50"]
# TYPE_LIST = ["base", "base_finetune", "meta", "meta_grad", "meta_grad_group_5", "meta_grad_group_10", "meta_grad_group_20", "meta_grad_group_30", "meta_grad_group_50"]

TYPE_LIST = ["base", "base_finetune", "meta",
             "meta_grad", "meta_grad_gru",
             "meta_grad_gru_center_group_2_clip_0.5", "meta_grad_gru_center_group_3_clip_0.5",
             "meta_grad_gru_center_group_5_clip_0.5", "meta_grad_gru_center_group_10_clip_0.5",
             "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
             "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
             ]
# NAME_LIST = ["Base", "DUET", "DUET (Grad.)", "DUET (Param.)", "DUET (Grad.)"]
# NAME_LIST = ["Base", "", "DUET", "DUET (Grad.)", "DUET (Grad.)-5", "DUET (Grad.)-10", "DUET (Grad.)-50", "DUET (Param.)", "DUET (Param.)-5", "DUET (Param.)-10", "DUET (Param.)-50", ]
# NAME_LIST = ["Base", "", "DUET", "DUET (Grad.)", "DUET (Grad.)-5", "DUET (Grad.)-10", "DUET (Grad.)-50"]
# NAME_LIST = ["Base", "Finetune", "DUET", "DUET (Grad.)", "DUET (Grad.)-5", "DUET (Grad.)-10", "DUET (Grad.)-20", "DUET (Grad.)-30", "DUET (Grad.)-50"]
NAME_LIST = ["Base", "Finetune", "DUET",
             "DUET (Grad.)", "DUET (Grad.GRU)",
             "DUET (Grad.GRU)-2-0.5", "DUET (Grad.GRU)-3-0.5",
             "DUET (Grad.GRU)-5-0.5", "DUET (Grad.GRU)-10-0.5",
             "DUET (Grad.GRU)-2-1.0", "DUET (Grad.GRU)-3-1.0",
             "DUET (Grad.GRU)-5-1.0", "DUET (Grad.GRU)-10-1.0",
             ]
# TYPE_LIST = ["base"]

# TYPE_LIST1 = ["base", "base_finetune", "meta", "meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood", "meta_ood_uncertainty"]
# TYPE_LIST2 = ["meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood", "meta_ood_uncertainty"]

# TYPE_LIST = ["base", "base_finetune", "meta", "meta_random", "meta_ood"]
# log_filename = "log.txt"
# log_filename = "log_ood.txt"
log_filename = "test.txt"
# epoch = 20
# epoch = 10

result_file = os.path.join(ROOT_FOLDER, "result_ood_overall.txt")
result_file2 = os.path.join(ROOT_FOLDER, "result2_ood_overall.txt")
# csv_writer1 = csv.writer(open(os.path.join(ROOT_FOLDER, "result_ood.csv"), 'w+', encoding='utf-8', newline=''))
result_csv1 = os.path.join(ROOT_FOLDER, "result_ood1_overall.csv")
result_csv2 = os.path.join(ROOT_FOLDER, "result_ood2_overall.csv")
# csv_writer2 = csv.writer(open(os.path.join(ROOT_FOLDER, "result_ood2.csv"), 'w+', encoding='utf-8', newline=''))
# with open(result_file, "w+") as writer, open(result_file2, "w+") as writer2:
with open(result_file, "w+") as writer, open(result_csv1, "w+") as csv_writer:
    for dataset in DATASET_LIST:
        print("=" * 50, file=writer)
        print(dataset, file=writer)
        print("-" * 50, file=writer)
        for model in MODEL_LIST:
            for (type, name) in zip(TYPE_LIST, NAME_LIST):
                root_folder = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type)
                log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
                # if type in ["base", "meta", "meta_grad", "meta_param"]:
                # if type in ["base", "meta", "meta_grad", "meta_grad_group_5", "meta_grad_group_10", "meta_grad_group_50",
                #             "meta_param", "meta_param_group_5", "meta_param_group_10", "meta_grad_group_20",
                #             "meta_grad_group_30", "meta_param_group_50"]:
                if type in ["base", "meta",
                            "meta_grad", "meta_grad_gru",
                            "meta_grad_gru_center_group_2_clip_0.5", "meta_grad_gru_center_group_3_clip_0.5",
                            "meta_grad_gru_center_group_5_clip_0.5", "meta_grad_gru_center_group_10_clip_0.5",
                            "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
                            "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
                            ]:
                    max_auc = 0
                    log_files = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_ood.txt")]
                elif type == "base_finetune":
                    log_files = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_overall.txt")]
                elif type == "2_prototype_long_tail":
                    # log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_overall.txt")
                    # log_files = [os.path.join(root_folder, file) for file in os.listdir(root_folder) if file.split(".")[-1] == "txt"]
                    log_files = [os.path.join(root_folder, "log_ood_5.txt"),
                                 os.path.join(root_folder, "log_ood_10.txt"),
                                 os.path.join(root_folder, "log_ood_50.txt"), ]
                    # for log_file in log_files:
                    #     log_files = [os.path.join(root_folder, log_file) for log_file in log_files]
                    # print(log_files)
                # if type(log_files) == "list":
                #     print(type(log_files))
                # log_files = sorted(log_files)
                for log_file in log_files:
                    if not os.path.exists(log_file):
                        continue
                    result_dict = defaultdict(list)
                    with open(log_file, "r+") as reader:
                        for index, line in enumerate(reader, 1):
                            # print(line.strip("\n"))
                            auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                            auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                            logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                            ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                            hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                            ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                            hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                            ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                            hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])

                            # if type == "base_finetune":
                            #     auc_max_list.append(auc)
                            #     auc_user_max_list.append(auc_user)
                            #     logloss_max_list.append(logloss)
                            #     ndcg5_max_list.append(ndcg5)
                            #     ndcg10_max_list.append(ndcg10)
                            #     ndcg20_max_list.append(ndcg20)
                            #     hr5_max_list.append(hr5)
                            #     hr10_max_list.append(hr10)
                            #     hr20_max_list.append(hr20)
                            #     continue

                            if type in ["base", "base_finetune"]:
                                rate = 0.0
                            # elif type in ["meta", "meta_grad", "meta_param"]:
                            # elif type in ["meta", "meta_grad", "meta_grad_group_5", "meta_grad_group_10", "meta_grad_group_50",
                            #               "2_prototype_long_tail", "meta_param", "meta_param_group_5", "meta_param_group_10",
                            #               "meta_grad_group_20", "meta_grad_group_30", "meta_param_group_50"]:
                            elif type in ["meta",
                                          "meta_grad", "meta_grad_gru",
                                          "meta_grad_gru_center_group_2_clip_0.5",
                                          "meta_grad_gru_center_group_3_clip_0.5",
                                          "meta_grad_gru_center_group_5_clip_0.5",
                                          "meta_grad_gru_center_group_10_clip_0.5",
                                          "meta_grad_gru_center_group_2_clip_1.0",
                                          "meta_grad_gru_center_group_3_clip_1.0",
                                          "meta_grad_gru_center_group_5_clip_1.0",
                                          "meta_grad_gru_center_group_10_clip_1.0",
                                          ]:
                                rate = 100.0
                            else:
                                # rate = float(line.strip("\n").split(",")[-4].split("=")[-1]) / float(line.strip("\n").split(",")[-3].split("=")[-1])
                                rate = float(line.strip("\n").split(",")[-1].split("=")[-1])
                                if rate == 0.0:
                                    continue
                            # if type in ["base", "meta", "meta_grad", "meta_param"]:
                            # if type in ["base", "meta", "meta_grad", "meta_grad_group_5", "meta_grad_group_10", "meta_grad_group_50",
                            #             "meta_param", "meta_param_group_5", "meta_param_group_10", "meta_grad_group_20",
                            #             "meta_grad_group_30", "meta_param_group_50"]:
                            if type in ["base", "meta",
                                        "meta_grad", "meta_grad_gru",
                                        "meta_grad_gru_center_group_2_clip_0.5",
                                        "meta_grad_gru_center_group_3_clip_0.5",
                                        "meta_grad_gru_center_group_5_clip_0.5",
                                        "meta_grad_gru_center_group_10_clip_0.5",
                                        "meta_grad_gru_center_group_2_clip_1.0",
                                        "meta_grad_gru_center_group_3_clip_1.0",
                                        "meta_grad_gru_center_group_5_clip_1.0",
                                        "meta_grad_gru_center_group_10_clip_1.0",
                                        ]:
                                # if auc > max_auc:
                                if auc_user > max_auc:
                                    # max_auc = auc
                                    max_auc = auc_user
                                    # result_dict[rate] = [auc, auc_user, ndcg10, hr10, ndcg20, hr20, rate]
                                    result_dict[rate] = [auc, auc_user, ndcg5, hr5, ndcg10, hr10, ndcg20, hr20, rate]
                                    continue
                                # break

                            if type in ["meta_ood_ocsvm", "meta_ood_lof"]:
                                request_num = float(line.strip("\n").split(",")[-5].split("=")[-1])
                                if request_num > 0:
                                    # auc, auc_user, ndcg10, hr10, ndcg20, hr20, rate = str(auc), str(auc_user), str(ndcg10), str(hr10), str(ndcg20), str(hr20), str(rate)
                                    result_dict[rate].extend(
                                        # [auc, auc_user, logloss, ndcg5, hr5, ndcg10, hr10, ndcg20, hr20, rate]
                                        # [auc, auc_user, logloss, ndcg10, hr10, ndcg20, hr20, rate]
                                        # [auc, auc_user, ndcg10, hr10, ndcg20, hr20, rate]
                                        # [auc, auc_user, ndcg10, hr10, ndcg20, hr20]
                                        [auc, auc_user, ndcg5, hr5, ndcg10, hr10, ndcg20, hr20]
                                    )
                                    break
                                else:
                                    continue
                            # auc, auc_user, ndcg10, hr10, ndcg20, hr20, rate = str(auc), str(auc_user), str(ndcg10), str(hr10), str(ndcg20), str(hr20), str(rate)
                            result_dict[rate].extend(
                                # [auc, auc_user, logloss, ndcg5, hr5, ndcg10, hr10, ndcg20, hr20, rate]
                                # [auc, auc_user, logloss, ndcg10, hr10, ndcg20, hr20, rate]
                                # [auc, auc_user, ndcg10, hr10, ndcg20, hr20, rate]
                                # [auc, auc_user, ndcg10, hr10, ndcg20, hr20]
                                [auc, auc_user, ndcg5, hr5, ndcg10, hr10, ndcg20, hr20]
                            )


                            # rate_list.append(rate)

                            # auc_dict[rate] = auc
                            # auc_user_dict[rate] = auc_user
                            # logloss_dict[rate] = logloss
                            # ndcg5_dict[rate] = ndcg5
                            # ndcg10_dict[rate] = ndcg10
                            # ndcg20_dict[rate] = ndcg20
                            # hr5_dict[rate] = hr5
                            # hr10_dict[rate] = hr10
                            # hr20_dict[rate] = hr20

                        result_dict = dict(sorted(result_dict.items(), key=lambda x: x[0]))

                        for key, value in result_dict.items():
                            # model, type = key.split(" ")
                            rate = key
                            value = list(map(str, value))
                            # if type == "base":
                            #     print(type)
                            print("=" * 100)
                            print(dataset)
                            print("-" * 50)
                            print(model)
                            print(type)
                            # print(result_dict)
                            print(key, value)
                            print("\t".join(value))
                            for _writer in [writer, csv_writer]:
                                # print("{}\t{}\t{}".format("{:10s}".format(model), "{:24s}{}".format(type, str(round(float(rate) * 100, 1)) + "%"),
                                # print("{}\t{}\t{}".format("{:10s}".format(model), "{:36s}{}".format(type, str(round(float(rate), 1)) + "%"),
                                #                           "\t".join(value)),
                                #                           sep="\t", file=_writer)
                                if type == "2_prototype_long_tail":
                                    group_num = log_file.split("/")[-1].split(".")[0].split("_")[-1]
                                    # print(group_num)
                                    # group_num = root_folder.split("/")[-1].split(".")[0].split("_")[-1]
                                    name = "PCGrad-{}".format(group_num)
                                #     print("{}\t{}\t{}".format("{:12s}".format(model), "{:36s}".format(type_),
                                #                               "\t".join(value)),
                                #           sep="\t", file=_writer)
                                # else:
                                #     print("{}\t{}\t{}".format("{:12s}".format(model), "{:36s}".format(type),
                                #                               "\t".join(value)),
                                #                               sep="\t", file=_writer)
                                print("{}\t{}\t{}".format("{:12s}".format(model), "{:36s}".format(name),
                                                          "\t".join(value)),
                                      sep="\t", file=_writer)
                                # print("{} {} {}".format("{:10s}".format(model), "{:36s}{}".format(type, str(round(float(rate), 1)) + "%"),
                                #                           " ".join(value)),
                                #       sep=" ", file=_writer)
                                # for metric_dict in [auc_dict, auc_user_dict, logloss_dict, ndcg5_dict, ndcg10_dict,
                                #                     ndcg20_dict, hr5_dict, hr10_dict, hr20_dict]:
                                #     for rate, metric in metric_dict.items():
                                #         print()
                                    # print("{},{},{},{},{},{},{},{},{}".format("{:10s}".format(model), "{:24s}".format(type),
                                    #                                           "{:6s}".format(",".join(result_dict["{} {}".format(model, type)])),
                                    #                                           "{:6s}".format(",".join(result_dict_["{} {}".format(model, type)])),
                                    #                                           "{:6s}".format(",".join(result_dict3["{} {}".format(model, type)])),
                                    #                                           "{:6s}".format(",".join(result_dict4["{} {}".format(model, type)])),
                                    #                                           "{:6s}".format(",".join(result_dict5["{} {}".format(model, type)])),
                                    #                                           "{:6s}".format(",".join(result_dict6["{} {}".format(model, type)])),
                                    #                                           "{:6s}".format(",".join(result_dict11["{} {}".format(model, type)]))),
                                    #       sep="\t", file=_writer)

        print("\n", file=writer)
        print("\n", file=csv_writer)
        # print("\n", file=writer2)

    # print("\n\n", file=writer)

