import sys

sys.path.append('../')

import collections
import os
import random
from pathlib import Path
import logging
import shutil
import time
from packaging import version
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import gzip
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import sys
sys.path.append("./")
sys.path.append("./src/")
sys.path.append("../")
sys.path.append("../src/")
from src.param import parse_args
from src.utils import LossMeter
from src.dist_utils import reduce_dict
from transformers import T5Tokenizer, T5TokenizerFast
from src.tokenization import P5Tokenizer, P5TokenizerFast
from src.pretrain_model import P5Pretraining

torch.set_num_threads(8)

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from src.trainer_base import TrainerBase

import pickle


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


import json


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


sys_args = sys.argv
# dataset = "beauty"
dataset = sys_args[1]
args = DotDict()

# args.framework = "small_recommendation_model"
args.framework = sys_args[2]
# args.small_recommendation_model = "sasrec"
args.small_recommendation_model = sys_args[3]
args.type = sys_args[4]
args.losses = sys_args[5]
# args.arch_config = "scripts/configs/amazon_{}_conf.json".format(dataset)
args.arch_config = "scripts/configs/{}_conf.json".format(dataset)
args.distributed = False
args.multiGPU = True
args.fp16 = True
args.train = dataset
args.valid = dataset
args.test = dataset
# args.batch_size = 4
# args.batch_size = 8
# args.batch_size = 16
# args.batch_size = 64
args.optim = 'adamw'
args.warmup_ratio = 0.05
args.lr = 1e-3
args.num_workers = 4
args.clip_grad_norm = 1.0
# args.losses = 'rating,sequential,explanation,review,traditional'
# args.losses = 'sequential'
# args.losses = ['sequential', "traditional"]
# args.losses = "sequential, traditional"
# args.losses = "sequential_ctr"
# args.losses = ["Sequential_ctr"]
# args.losses = sys_args[4]
# args.backbone = 't5-small'  # small or base
# args.backbone = '/data/lvzheqi/lvzheqi/ICLR2023/LLM/llm/t5-small'  # small or base
args.backbone = '/data/home/lzq/lvzheqi/ICLR2023/LLM/llm/t5-small'  # small or base
args.model_name = 't5-small'
# args.output = 'snap/beauty-small'
# args.epoch = 10
args.local_rank = 0

args.comment = ''
args.train_topk = -1
args.valid_topk = -1
args.dropout = 0.1

args.tokenizer = 'p5'
args.max_text_length = 512
args.do_lower_case = False
args.word_mask_rate = 0.15
args.gen_max_length = 64

args.weight_decay = 0.01
args.adam_eps = 1e-6
args.gradient_accumulation_steps = 1

'''
Set seeds
'''
args.seed = 2022
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

'''
Whole word embedding
'''
args.whole_word_embed = True

cudnn.benchmark = True
ngpus_per_node = torch.cuda.device_count()
args.world_size = ngpus_per_node

LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
if args.local_rank in [0, -1]:
    print(LOSSES_NAME)
LOSSES_NAME.append('total_loss')  # total loss

args.LOSSES_NAME = LOSSES_NAME

gpu = 0  # Change GPU ID
args.gpu = gpu
args.rank = gpu
print(f'Process Launching at GPU {gpu}')

torch.cuda.set_device('cuda:{}'.format(gpu))

# comments = []
# dsets = []
# # if 'toys' in args.train:
# #     dsets.append('toys')
# # if 'beauty' in args.train:
# #     dsets.append('beauty')
# # if 'sports' in args.train:
# #     dsets.append('sports')
# dsets.append(dataset)
# comments.append(''.join(dsets))
# if args.backbone:
#     comments.append(args.backbone)
# comments.append(''.join(args.losses.split(',')))
# if args.comment != '':
#     comments.append(args.comment)
# comment = '_'.join(comments)
#
# if args.local_rank in [0, -1]:
#     print(args)
#
#
# def create_config(args):
#     from transformers import T5Config, BartConfig
#
#     if 't5' in args.backbone:
#         config_class = T5Config
#     else:
#         return None
#
#     config = config_class.from_pretrained(args.backbone)
#     config.dropout_rate = args.dropout
#     config.dropout = args.dropout
#     config.attention_dropout = args.dropout
#     config.activation_dropout = args.dropout
#     config.losses = args.losses
#
#     return config
#
#
# def create_tokenizer(args):
#     from transformers import T5Tokenizer, T5TokenizerFast
#     from src.tokenization import P5Tokenizer, P5TokenizerFast
#
#     if 'p5' in args.tokenizer:
#         tokenizer_class = P5Tokenizer
#
#     tokenizer_name = args.backbone
#
#     tokenizer = tokenizer_class.from_pretrained(
#         tokenizer_name,
#         max_length=args.max_text_length,
#         do_lower_case=args.do_lower_case,
#     )
#
#     print(tokenizer_class, tokenizer_name)
#
#     return tokenizer
#
#
# def create_model(model_class, config=None):
#     print(f'Building Model at GPU {args.gpu}')
#
#     model_name = args.backbone
#
#     model = model_class.from_pretrained(
#         model_name,
#         config=config
#     )
#     return model
#
#
# config = create_config(args)
#
# if args.tokenizer is None:
#     args.tokenizer = args.backbone
#
# tokenizer = create_tokenizer(args)
#
# model_class = P5Pretraining
# model = create_model(model_class, config)
#
# model = model.cuda()
#
# if 'p5' in args.tokenizer:
#     model.resize_token_embeddings(tokenizer.vocab_size)
#
# model.tokenizer = tokenizer
#
# # args.load = "../snap/beauty-small.pth"
# # args.load = "snap/beauty-small.pth"
# # args.load = "beauty/BEST_EVAL_LOSS.pth"
# # args.load = "{}/{}/BEST_EVAL_LOSS.pth".format(dataset, args.model_name)
#
# # Load Checkpoint
from src.utils import load_state_dict, LossMeter, set_global_logging_level
# from pprint import pprint
#
# def load_checkpoint(ckpt_path):
#     state_dict = load_state_dict(ckpt_path, 'cpu')
#     results = model.load_state_dict(state_dict, strict=False)
#     print('Model loaded from ', ckpt_path)
#     pprint(results)
#
# # ckpt_path = args.load
# # load_checkpoint(ckpt_path)
#
# from src.all_amazon_templates import all_tasks as task_templates
#
# # data_splits = load_pickle('../data/beauty/rating_splits_augmented.pkl')
# data_splits = load_pickle('data/{}/rating_splits_augmented.pkl'.format(dataset))
# test_review_data = data_splits['test']
#
# print(len(test_review_data))
#
# print(test_review_data[0])
#
# # data_maps = load_json(os.path.join('../data', 'beauty', 'datamaps.json'))
# data_maps = load_json(os.path.join('data', dataset, 'datamaps.json'))
# print(len(data_maps['user2id'])) # number of users
# print(len(data_maps['item2id'])) # number of items
#
#
# from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data import get_loader
# from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
# from evaluate.metrics4rec import evaluate_all

# print("====================rating====================")
# test_task_list = {'rating': ['1-10']  # or '1-6'
#                   }
# test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
#
# zeroshot_test_loader = get_loader(
#     args,
#     test_task_list,
#     test_sample_numbers,
#     split=args.test,
#     mode='test',
#     batch_size=args.batch_size,
#     workers=args.num_workers,
#     distributed=args.distributed
# )
# print(len(zeroshot_test_loader))
#
# gt_ratings = []
# pred_ratings = []
# for i, batch in tqdm(enumerate(zeroshot_test_loader)):
#     with torch.no_grad():
#         results = model.generate_step(batch)
#         gt_ratings.extend(batch['target_text'])
#         pred_ratings.extend(results)
#
# predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if
#                     p in [str(i / 10.0) for i in list(range(10, 50))]]
# RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
# print('RMSE {:7.4f}'.format(RMSE))
# MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
# print('MAE {:7.4f}'.format(MAE))
#
# test_task_list = {'rating': ['1-6']  # or '1-10'
#                   }
# test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
#
# zeroshot_test_loader = get_loader(
#     args,
#     test_task_list,
#     test_sample_numbers,
#     split=args.test,
#     mode='test',
#     batch_size=args.batch_size,
#     workers=args.num_workers,
#     distributed=args.distributed
# )
# print(len(zeroshot_test_loader))
#
# gt_ratings = []
# pred_ratings = []
# for i, batch in tqdm(enumerate(zeroshot_test_loader)):
#     with torch.no_grad():
#         results = model.generate_step(batch)
#         gt_ratings.extend(batch['target_text'])
#         pred_ratings.extend(results)
#
# predicted_rating = [(float(r), float(p)) for (r, p) in zip(gt_ratings, pred_ratings) if
#                     p in [str(i / 10.0) for i in list(range(10, 50))]]
# RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
# print('RMSE {:7.4f}'.format(RMSE))
# MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
# print('MAE {:7.4f}'.format(MAE))
args.losses = args.losses.split(",")

print("args.losses ", args.losses)
for loss_ in args.losses:
    if loss_ == "Sequential_ctr":
        args.batch_size = 8
    elif loss_ == "Traditional_ctr":
        args.batch_size = 64
    else:
        raise NotImplementedError
    print("loss_ ", loss_)
    if loss_ is None:
        continue
    save_dir = "evaluate_small_new_metrics_"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # print(os.path.join(save_dir, "{}_{}_{}.txt".format(dataset, args.small_recommendation_model, loss_)))
    with open(os.path.join(save_dir, "{}_{}_{}_{}.txt".format(dataset, args.type, args.small_recommendation_model, loss_)), "w+") as writer:
        # for epoch in range(10):
        # for epoch in range(7, 10):
        for epoch in range(3):
            # train_epoch = epoch + 1
            train_epoch = 10 - epoch
            # args.load = "checkpoint/{}/{}_{}/BEST_EVAL_LOSS.pth".format(dataset, args.model_name, loss_)
            # args.load = "checkpoint/{}/{}_{}/Epoch{}.pth".format(dataset, args.type, args.small_recommendation_model, str(train_epoch).zfill(2))
            args.load = "checkpoint/{}/{}_{}_{}/Epoch{}.pth".format(dataset, args.type, args.small_recommendation_model, loss_, str(train_epoch).zfill(2))
            ckpt_path = args.load
            print("ckpt_path ", ckpt_path)

            import sys

            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from util import consts
            import model
            from util import args_processing as ap

            if args.framework == "small_recommendation_model_base":
                model_meta = model.get_model_meta(args.small_recommendation_model)  # type: model.ModelMeta
                model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

                # Construct model
                small_recommendation_model = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
                # # load_checkpoint(ckpt_path)
                # small_recommendation_model = torch.load(ckpt_path)
                state_dict = load_state_dict(ckpt_path, 'cpu')
                # results = model.load_state_dict(state_dict, strict=False)
                small_recommendation_model.load_state_dict(state_dict, strict=False)

            elif args.framework == "small_recommendation_model_duet":
                small_recommendation_model = torch.load(ckpt_path)
            # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict
            # print(small_recommendation_model)
            LOSSES_NAME = args.LOSSES_NAME
            print("-" * 50)

            # if args.dry:
            #     results = evaluate_epoch(epoch=0)

            # if verbose:
            #     loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
            #     best_eval_loss = 100000.
            #
            #     if 't5' in args.backbone:
            #         project_name = "P5_Pretrain"
            #
            #     src_dir = Path(__file__).resolve().parent
            #     base_path = str(src_dir.parent)
            #     src_dir = str(src_dir)

            if args.distributed:
                dist.barrier()

            global_step = 0
            # optimizer = torch.optim.Adam(
            #     small_recommendation_model.parameters(),
            #     # lr=float(args.learning_rate)
            #     lr=float(args.lr)
            # )
            # small_recommendation_model.train()
            from util import env

            device = env.get_device()
            small_recommendation_model.to(device)
            # start_epoch = None
            # LOSSES_NAME = self.args.LOSSES_NAME
            LOSSES_NAME = args.LOSSES_NAME

            # epoch_results = {}
            # for loss_name in LOSSES_NAME:
            #     epoch_results[loss_name] = 0.
            #     epoch_results[f'{loss_name}_count'] = 0

            # self.large_language_model.eval()
            # results = {}
            # valid_results = {}
            # self.small_recommendation_model.eval()
            # valid_loss = 0
            # valid_line_count = 0
            # val_loader = get_loader(
            #     args,
            #     val_task_list,
            #     val_sample_numbers,
            #     split=args.valid,
            #     mode='val',
            #     batch_size=args.batch_size,
            #     workers=args.num_workers,
            #     distributed=args.distributed
            # )
            # test_task_list = {'sequential': ['2-13']  # or '2-3'
            #                   }
            # test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            test_task_list = {'Sequential_ctr': []}
            # define sampling numbers for each group of personalized prompts (see pretrain_data.py)
            # if greater than 1, a data sample will be used for multiple times with different prompts in certain task family
            # train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1, 'traditional': (10, 5)}
            test_sample_numbers = {'Sequential_ctr': (5, 5, 10)}
            test_loader = get_loader(
                args,
                test_task_list,
                test_sample_numbers,
                split=args.test,
                mode='test',
                batch_size=args.batch_size,
                workers=args.num_workers,
                distributed=args.distributed
            )
            from util import new_metrics
            pred_list = []
            y_list = []
            buffer = []
            import logging
            import time
            from util.timer import Timer
            timer = Timer()
            user_id_list = []
            log_every = 200
            logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
            logger = logging.getLogger(__name__)
            # train_epoch = 0
            train_step = 0
            with torch.no_grad():
                # if self.verbose:
                #     loss_meter = LossMeter()
                #     loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
                #
                #     pbar = tqdm(total=len(self.val_loader), ncols=275)

                # for step_i, batch in enumerate(self.val_loader):
                #
                #     if self.args.distributed:
                #         results = self.large_language_model.module.valid_step(batch)
                #     else:
                #         results = self.large_language_model.valid_step(batch)
                # for step_i, batch in tqdm(enumerate(self.test_loader)):
                for step_i, batch in tqdm(enumerate(test_loader)):
                    logits = small_recommendation_model({
                        key: value.to(device)
                        for key, value in batch.items()
                        # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                        if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE, consts.FIELD_TRIGGER_SEQUENCE}
                    })
                    # criterion = nn.BCEWithLogitsLoss()
                    # loss = criterion(logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))

                    prob = torch.sigmoid(logits).detach().cpu().numpy()
                    y = batch[consts.FIELD_LABEL].view(-1, 1)
                    # fpr, tpr, thresholds = new_metrics.roc_curve(np.array(y), prob, pos_label=1)
                    # overall_auc = float(new_metrics.overall_auc(fpr, tpr))
                    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
                    # ndcg = 0
                    user_id_list.extend(np.array(batch[consts.FIELD_USER_ID].view(-1, 1)))
                    pred_list.extend(prob)
                    y_list.extend(np.array(y))
                    # for user_id, score, target, sequence, label in zip(
                    #         batch[consts.FIELD_USER_ID],
                    #         prob, batch[consts.FIELD_TARGET_ID], batch[consts.FIELD_CLK_SEQUENCE],
                    #         batch[consts.FIELD_LABEL]
                    #     ):
                    #     print("=" * 50)
                    #     print(user_id)
                    #     print(score)
                    #     print(target)
                    #     print(sequence)
                    #     print(label)
                    buffer.extend(
                        # [str(user_id), float(score), float(label)]
                        [int(user_id), float(score), float(label)]
                        for user_id, score, label
                        in zip(
                            batch[consts.FIELD_USER_ID],
                            prob,
                            batch[consts.FIELD_LABEL]
                        )
                    )
                    # if step % log_every == 0:
                    if step_i % log_every == 0:
                        logger.info(
                            "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
                                # train_epoch, step, overall_auc, log_every / timer.tick(False)
                                train_epoch, step_i, overall_auc, log_every / timer.tick(False)
                            )
                        )
                    # valid_loss += loss.detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                    # valid_line_count += len(batch[consts.FIELD_LABEL].view(-1, 1))
                    # for k, v in results.items():
                    #     if k in epoch_results:
                    #         if isinstance(v, int):
                    #             epoch_results[k] += v
                    #         elif isinstance(v, torch.Tensor):
                    #             epoch_results[k] += v.item()

                    # if self.verbose and step_i % 200:
                    #     desc_str = f'Valid Epoch {epoch} |'
                    #     # for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                    #     #
                    #     #     if loss_name in results:
                    #     #         # loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                    #     #         loss_meter.update(loss)
                    #     #     if len(loss_meter) > 0:
                    #     #         loss_count = epoch_results[f'{loss_name}_count']
                    #     #         desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                    #     desc_str += f' {self.args.losses}_loss ({valid_line_count}) {loss_meter.val:.3f}'
                    #
                    #     pbar.set_description(desc_str)
                    #     pbar.update(1)
                    # dist.barrier()

                # valid_results["total_loss"] = valid_loss
                # valid_results["total_loss_count"] = valid_line_count

                # if self.verbose:
                #     pbar.close()
                # dist.barrier()

                # return epoch_results
                # return valid_loss
                # return valid_results
            overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
            user_auc = new_metrics.calculate_user_auc(buffer)
            overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
            # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
            user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
            user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
            user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
            avg_prec5, mrr5 = new_metrics.calculate_user_prec_mrr(5, buffer)
            avg_prec10, mrr10 = new_metrics.calculate_user_prec_mrr(10, buffer)
            avg_prec20, mrr20 = new_metrics.calculate_user_prec_mrr(20, buffer)
            # avg_prec = avg_prec / cnt
            # avg_recall = avg_recall / cnt
            # avg_ndcg = avg_ndcg / cnt
            # avg_hit = avg_hit / cnt
            # map_ = mean_average_precision(rs)
            # mrr = mean_reciprocal_rank(rs)
            # ndcg = 0
            # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
            #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
            #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
            # # with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
            # # with open(os.path.join(save_dir, "{}_{}_{}.txt".format(dataset, args.small_recommendation_model, loss_)), "a+") as writer:
            # #     print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            # #           "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
            # #           format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
            # #                  user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
            # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
            #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
            #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
            # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
            #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
            #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
            # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            #       "user_ndcg5={:5f}, map5={:5f}, recall5={:5f}, precision5={:5f}, mrr5={:5f}, user_hr5={:5f}, "
            #       "user_ndcg10={:5f}, map10={:5f}, recall10={:5f}, precision10={:5f}, mrr10={:5f}, user_hr10={:5f}, "
            #       "user_ndcg20={:5f}, map20={:5f}, recall20={:5f}, precision20={:5f}, mrr20={:5f}, user_hr20={:5f}".
            #       format(
            #     train_epoch, train_step, overall_auc, user_auc, overall_logloss,
            #     user_ndcg5, map5, recall5, precision5, mrr5, user_hr5,
            #     user_ndcg10, map10, recall10, precision10, mrr10, user_hr10,
            #     user_ndcg20, map20, recall20, precision20, mrr20, user_hr20,
            # ))
            print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
                  "user_ndcg5={:5f}, user_hr5={:5f}, precision5={:5f}, mrr5={:5f}, "
                  "user_ndcg10={:5f}, user_hr10={:5f}, precision10={:5f}, mrr10={:5f}, "
                  "user_ndcg20={:5f}, user_hr20={:5f}, precision20={:5f}, mrr20={:5f}".
            format(
                train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                user_ndcg5, user_hr5, avg_prec5, mrr5,
                user_ndcg10, user_hr10, avg_prec10, mrr10,
                user_ndcg20, user_hr20, avg_prec20, mrr20,
            ))
            # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
            #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
            #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
            # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
            #       "user_ndcg5={:5f}, map5={:5f}, recall5={:5f}, precision5={:5f}, mrr5={:5f}, user_hr5={:5f}, "
            #       "user_ndcg10={:5f}, map10={:5f}, recall10={:5f}, precision10={:5f}, mrr10={:5f}, user_hr10={:5f}, "
            #       "user_ndcg20={:5f}, map20={:5f}, recall20={:5f}, precision20={:5f}, mrr20={:5f}, user_hr20={:5f}".
            # format(
            #     train_epoch, train_step, overall_auc, user_auc, overall_logloss,
            #     user_ndcg5, map5, recall5, precision5, mrr5, user_hr5,
            #     user_ndcg10, map10, recall10, precision10, mrr10, user_hr10,
            #     user_ndcg20, map20, recall20, precision20, mrr20, user_hr20,
            # ), file=writer)
            print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
                  "user_ndcg5={:5f}, user_hr5={:5f}, precision5={:5f}, mrr5={:5f}, "
                  "user_ndcg10={:5f}, user_hr10={:5f}, precision10={:5f}, mrr10={:5f}, "
                  "user_ndcg20={:5f}, user_hr20={:5f}, precision20={:5f}, mrr20={:5f}".
            format(
                train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                user_ndcg5, user_hr5, avg_prec5, mrr5,
                user_ndcg10, user_hr10, avg_prec10, mrr10,
                user_ndcg20, user_hr20, avg_prec20, mrr20,
            ), file=writer)
            # # if args.losses == "sequential":
            # if loss_ == "sequential":
            #     print("====================sequential====================")
            #     test_task_list = {'sequential': ['2-13']  # or '2-3'
            #                       }
            #     test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            #
            #     zeroshot_test_loader = get_loader(
            #         args,
            #         test_task_list,
            #         test_sample_numbers,
            #         split=args.test,
            #         mode='test',
            #         batch_size=args.batch_size,
            #         workers=args.num_workers,
            #         distributed=args.distributed
            #     )
            #     print(len(zeroshot_test_loader))
            #
            #     all_info = []
            #     for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            #         with torch.no_grad():
            #             results = model.generate_step(batch)
            #             beam_outputs = model.generate(
            #                 batch['input_ids'].to('cuda'),
            #                 max_length=50,
            #                 num_beams=20,
            #                 no_repeat_ngram_size=0,
            #                 num_return_sequences=20,
            #                 early_stopping=True
            #             )
            #             generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            #             for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            #                 new_info = {}
            #                 new_info['target_item'] = item[1]
            #                 new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
            #                 all_info.append(new_info)
            #
            #     gt = {}
            #     ui_scores = {}
            #     for i, info in enumerate(all_info):
            #         gt[i] = [int(info['target_item'])]
            #         pred_dict = {}
            #         for j in range(len(info['gen_item_list'])):
            #             try:
            #                 pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
            #             except:
            #                 pass
            #         ui_scores[i] = pred_dict
            #
            #     for num in [1, 5, 10, 20]:
            #         evaluate_results = evaluate_all(ui_scores, gt, num)
            #         # with open("{}_{}.txt".format(dataset, args.losses), "a+") as writer:
            #         with open("{}_{}.txt".format(dataset, loss_), "a+") as writer:
            #             print(test_task_list, num, file=writer)
            #             print(evaluate_results, file=writer)
            #     # evaluate_all(ui_scores, gt, 5)
            #     # evaluate_all(ui_scores, gt, 10)
            #
            #     test_task_list = {'sequential': ['2-3']  # or '2-13'
            #                       }
            #     test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            #
            #     zeroshot_test_loader = get_loader(
            #         args,
            #         test_task_list,
            #         test_sample_numbers,
            #         split=args.test,
            #         mode='test',
            #         batch_size=args.batch_size,
            #         workers=args.num_workers,
            #         distributed=args.distributed
            #     )
            #     print(len(zeroshot_test_loader))
            #
            #     all_info = []
            #     for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            #         with torch.no_grad():
            #             results = model.generate_step(batch)
            #             beam_outputs = model.generate(
            #                 batch['input_ids'].to('cuda'),
            #                 max_length=50,
            #                 num_beams=20,
            #                 no_repeat_ngram_size=0,
            #                 num_return_sequences=20,
            #                 early_stopping=True
            #             )
            #             generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            #             for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            #                 new_info = {}
            #                 new_info['target_item'] = item[1]
            #                 new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
            #                 all_info.append(new_info)
            #
            #     gt = {}
            #     ui_scores = {}
            #     for i, info in enumerate(all_info):
            #         gt[i] = [int(info['target_item'])]
            #         pred_dict = {}
            #         for j in range(len(info['gen_item_list'])):
            #             try:
            #                 pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
            #             except:
            #                 pass
            #         ui_scores[i] = pred_dict
            #
            #     for num in [1, 5, 10, 20]:
            #         evaluate_results = evaluate_all(ui_scores, gt, num)
            #         # with open("{}_{}.txt".format(dataset, args.losses), "a+") as writer:
            #         with open("{}_{}.txt".format(dataset, loss_), "a+") as writer:
            #             print(test_task_list, num, file=writer)
            #             print(evaluate_results, file=writer)
            #     # evaluate_all(ui_scores, gt, 1)
            #     # evaluate_all(ui_scores, gt, 5)
            #     # evaluate_all(ui_scores, gt, 10)
            #     # evaluate_all(ui_scores, gt, 20)
            #
            # # print("====================explanation====================")
            # # test_task_list = {'explanation': ['3-12']  # or '3-9' or '3-3'
            # #                   }
            # # test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            # #
            # # zeroshot_test_loader = get_loader(
            # #     args,
            # #     test_task_list,
            # #     test_sample_numbers,
            # #     split=args.test,
            # #     mode='test',
            # #     batch_size=args.batch_size,
            # #     workers=args.num_workers,
            # #     distributed=args.distributed
            # # )
            # # print(len(zeroshot_test_loader))
            # #
            # # tokens_predict = []
            # # tokens_test = []
            # # for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            # #     with torch.no_grad():
            # #         outputs = model.generate(
            # #             batch['input_ids'].to('cuda'),
            # #             min_length=9,
            # #             num_beams=12,
            # #             num_return_sequences=1,
            # #             num_beam_groups=3,
            # #             repetition_penalty=0.7
            # #         )
            # #         results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # #         tokens_predict.extend(results)
            # #         tokens_test.extend(batch['target_text'])
            # #
            # # new_tokens_predict = [l.split() for l in tokens_predict]
            # # new_tokens_test = [ll.split() for ll in tokens_test]
            # # BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
            # # BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
            # # ROUGE = rouge_score(tokens_test, tokens_predict)
            # #
            # # print('BLEU-1 {:7.4f}'.format(BLEU1))
            # # print('BLEU-4 {:7.4f}'.format(BLEU4))
            # # for (k, v) in ROUGE.items():
            # #     print('{} {:7.4f}'.format(k, v))
            # #
            # # test_task_list = {'explanation': ['3-9']  # or '3-12' or '3-3'
            # #                   }
            # # test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            # #
            # # zeroshot_test_loader = get_loader(
            # #     args,
            # #     test_task_list,
            # #     test_sample_numbers,
            # #     split=args.test,
            # #     mode='test',
            # #     batch_size=args.batch_size,
            # #     workers=args.num_workers,
            # #     distributed=args.distributed
            # # )
            # # print(len(zeroshot_test_loader))
            # #
            # # tokens_predict = []
            # # tokens_test = []
            # # for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            # #     with torch.no_grad():
            # #         outputs = model.generate(
            # #             batch['input_ids'].to('cuda'),
            # #             min_length=10,
            # #             num_beams=12,
            # #             num_return_sequences=1,
            # #             num_beam_groups=3
            # #         )
            # #         results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # #         tokens_predict.extend(results)
            # #         tokens_test.extend(batch['target_text'])
            # #
            # # new_tokens_predict = [l.split() for l in tokens_predict]
            # # new_tokens_test = [ll.split() for ll in tokens_test]
            # # BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
            # # BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
            # # ROUGE = rouge_score(tokens_test, tokens_predict)
            # #
            # # print('BLEU-1 {:7.4f}'.format(BLEU1))
            # # print('BLEU-4 {:7.4f}'.format(BLEU4))
            # # for (k, v) in ROUGE.items():
            # #     print('{} {:7.4f}'.format(k, v))
            # #
            # # test_task_list = {'explanation': ['3-3']  # or '3-12' or '3-9'
            # #                   }
            # # test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            # #
            # # zeroshot_test_loader = get_loader(
            # #     args,
            # #     test_task_list,
            # #     test_sample_numbers,
            # #     split=args.test,
            # #     mode='test',
            # #     batch_size=args.batch_size,
            # #     workers=args.num_workers,
            # #     distributed=args.distributed
            # # )
            # # print(len(zeroshot_test_loader))
            # #
            # # tokens_predict = []
            # # tokens_test = []
            # # for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            # #     with torch.no_grad():
            # #         outputs = model.generate(
            # #             batch['input_ids'].to('cuda'),
            # #             min_length=10
            # #         )
            # #         results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # #         tokens_predict.extend(results)
            # #         tokens_test.extend(batch['target_text'])
            # #
            # # new_tokens_predict = [l.split() for l in tokens_predict]
            # # new_tokens_test = [ll.split() for ll in tokens_test]
            # # BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
            # # BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
            # # ROUGE = rouge_score(tokens_test, tokens_predict)
            # #
            # # print('BLEU-1 {:7.4f}'.format(BLEU1))
            # # print('BLEU-4 {:7.4f}'.format(BLEU4))
            # # for (k, v) in ROUGE.items():
            # #     print('{} {:7.4f}'.format(k, v))
            #
            # # print("====================review====================")
            # # test_task_list = {'review': ['4-4']  # or '4-2'
            # #                   }
            # # test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            # #
            # # zeroshot_test_loader = get_loader(
            # #     args,
            # #     test_task_list,
            # #     test_sample_numbers,
            # #     split=args.test,
            # #     mode='test',
            # #     batch_size=args.batch_size,
            # #     workers=args.num_workers,
            # #     distributed=args.distributed
            # # )
            # # print(len(zeroshot_test_loader))
            # #
            # # gt_ratings = []
            # # pred_ratings = []
            # # for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            # #     if i > 50:
            # #         break
            # #     with torch.no_grad():
            # #         results = model.generate_step(batch)
            # #         gt_ratings.extend(batch['target_text'])
            # #         pred_ratings.extend(results)
            # #
            # # predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
            # # RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
            # # print('RMSE {:7.4f}'.format(RMSE))
            # # MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
            # # print('MAE {:7.4f}'.format(MAE))
            # #
            # # test_task_list = {'review': ['4-2']  # or '4-4'
            # #                   }
            # # test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            # #
            # # zeroshot_test_loader = get_loader(
            # #     args,
            # #     test_task_list,
            # #     test_sample_numbers,
            # #     split=args.test,
            # #     mode='test',
            # #     batch_size=args.batch_size,
            # #     workers=args.num_workers,
            # #     distributed=args.distributed
            # # )
            # # print(len(zeroshot_test_loader))
            # #
            # # gt_ratings = []
            # # pred_ratings = []
            # # for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            # #     if i > 50:
            # #         break
            # #     with torch.no_grad():
            # #         results = model.generate_step(batch)
            # #         gt_ratings.extend(batch['target_text'])
            # #         pred_ratings.extend(results)
            # #
            # # predicted_rating = [(float(r), round(float(p))) for (r, p) in zip(gt_ratings, pred_ratings)]
            # # RMSE = root_mean_square_error(predicted_rating, 5.0, 1.0)
            # # print('RMSE {:7.4f}'.format(RMSE))
            # # MAE = mean_absolute_error(predicted_rating, 5.0, 1.0)
            # # print('MAE {:7.4f}'.format(MAE))
            # #
            # # test_task_list = {'review': ['4-1']
            # #                   }
            # # test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            # #
            # # zeroshot_test_loader = get_loader(
            # #     args,
            # #     test_task_list,
            # #     test_sample_numbers,
            # #     split=args.test,
            # #     mode='test',
            # #     batch_size=args.batch_size,
            # #     workers=args.num_workers,
            # #     distributed=args.distributed
            # # )
            # # print(len(zeroshot_test_loader))
            # #
            # # tokens_predict = []
            # # tokens_test = []
            # # for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            # #     if i > 50:
            # #         break
            # #     with torch.no_grad():
            # #         results = model.generate_step(batch)
            # #         tokens_predict.extend(results)
            # #         tokens_test.extend(batch['target_text'])
            # #
            # # new_tokens_predict = [l.split() for l in tokens_predict]
            # # new_tokens_test = [ll.split() for ll in tokens_test]
            # # BLEU2 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=2, smooth=False)
            # # ROUGE = rouge_score(tokens_test, tokens_predict)
            # #
            # # print('BLEU-2 {:7.4f}'.format(BLEU2))
            # # for (k, v) in ROUGE.items():
            # #     print('{} {:7.4f}'.format(k, v))
            # # elif args.losses == "traditional":
            # elif loss_ == "traditional":
            #     print("====================traditional====================")
            #     test_task_list = {'traditional': ['5-8']  # or '5-5'
            #                       }
            #     test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            #
            #     zeroshot_test_loader = get_loader(
            #         args,
            #         test_task_list,
            #         test_sample_numbers,
            #         split=args.test,
            #         mode='test',
            #         batch_size=args.batch_size,
            #         workers=args.num_workers,
            #         distributed=args.distributed
            #     )
            #     print(len(zeroshot_test_loader))
            #
            #     all_info = []
            #     for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            #         with torch.no_grad():
            #             results = model.generate_step(batch)
            #             beam_outputs = model.generate(
            #                 batch['input_ids'].to('cuda'),
            #                 max_length=50,
            #                 num_beams=20,
            #                 no_repeat_ngram_size=0,
            #                 num_return_sequences=20,
            #                 early_stopping=True
            #             )
            #             generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            #             for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            #                 new_info = {}
            #                 new_info['target_item'] = item[1]
            #                 new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
            #                 all_info.append(new_info)
            #
            #     gt = {}
            #     ui_scores = {}
            #     for i, info in enumerate(all_info):
            #         gt[i] = [int(info['target_item'])]
            #         pred_dict = {}
            #         for j in range(len(info['gen_item_list'])):
            #             try:
            #                 pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
            #             except:
            #                 pass
            #         ui_scores[i] = pred_dict
            #
            #     for num in [1, 5, 10, 20]:
            #         evaluate_results = evaluate_all(ui_scores, gt, num)
            #         # with open("{}_{}.txt".format(dataset, args.losses), "a+") as writer:
            #         with open("{}_{}.txt".format(dataset, loss_), "a+") as writer:
            #             print(test_task_list, num, file=writer)
            #             print(evaluate_results, file=writer)
            #     # evaluate_all(ui_scores, gt, 1)
            #     # evaluate_all(ui_scores, gt, 5)
            #     # evaluate_all(ui_scores, gt, 10)
            #
            #     test_task_list = {'traditional': ['5-5']  # or '5-8'
            #                       }
            #     test_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
            #
            #     zeroshot_test_loader = get_loader(
            #         args,
            #         test_task_list,
            #         test_sample_numbers,
            #         split=args.test,
            #         mode='test',
            #         batch_size=args.batch_size,
            #         workers=args.num_workers,
            #         distributed=args.distributed
            #     )
            #     print(len(zeroshot_test_loader))
            #
            #     all_info = []
            #     for i, batch in tqdm(enumerate(zeroshot_test_loader)):
            #         with torch.no_grad():
            #             results = model.generate_step(batch)
            #             beam_outputs = model.generate(
            #                 batch['input_ids'].to('cuda'),
            #                 max_length=50,
            #                 num_beams=20,
            #                 no_repeat_ngram_size=0,
            #                 num_return_sequences=20,
            #                 early_stopping=True
            #             )
            #             generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            #             for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            #                 new_info = {}
            #                 new_info['target_item'] = item[1]
            #                 new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
            #                 all_info.append(new_info)
            #
            #     gt = {}
            #     ui_scores = {}
            #     for i, info in enumerate(all_info):
            #         gt[i] = [int(info['target_item'])]
            #         pred_dict = {}
            #         for j in range(len(info['gen_item_list'])):
            #             try:
            #                 pred_dict[int(info['gen_item_list'][j])] = -(j + 1)
            #             except:
            #                 pass
            #         ui_scores[i] = pred_dict
            #
            #     for num in [1, 5, 10, 20]:
            #         evaluate_results = evaluate_all(ui_scores, gt, num)
            #         # with open("{}_{}.txt".format(dataset, args.losses), "a+") as writer:
            #         with open("{}_{}.txt".format(dataset, loss_), "a+") as writer:
            #             print(test_task_list, num, file=writer)
            #             print(evaluate_results, file=writer)
            #
            #     # evaluate_all(ui_scores, gt, 1)
            #     # evaluate_all(ui_scores, gt, 5)
            #     # evaluate_all(ui_scores, gt, 10)
            #     # evaluate_all(ui_scores, gt, 20)
