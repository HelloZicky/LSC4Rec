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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from src.param import parse_args
from src.utils import LossMeter
from src.dist_utils import reduce_dict
from transformers import T5Tokenizer, T5TokenizerFast
from src.tokenization import P5Tokenizer, P5TokenizerFast
from src.pretrain_model import P5Pretraining
from torch.utils.data import DataLoader, Dataset, Sampler
from src.pretrain_data_dcc_t2 import get_loader
from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from evaluate.metrics4rec import evaluate_all



torch.set_num_threads(8)

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from src.trainer_base import TrainerBase

import pickle

from util import new_metrics
import logging
import time
from util.timer import Timer
timer = Timer()

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

#-----------------------------------------------------------can be modified
dataset = "beauty"
gpu0 = 1
request_ratio = 5
request_mode = 'Random' #Random or Colla
score_limit = 31.653333333333336 #please first run Random and check the output file get the ratio-score_limit then you can run Colla
save_dir = "test_result/dcc_t2/random"


args = DotDict()
args.backbone = '/data/zhantianyu/LLM/llm/t5-small'  # small or base
print(args.backbone)
args.dcc = True # True is using dcc data; False is using real-time data for llm
args.batch_size = 16
args.small_recommendation_model = 'sasrec'
args.collaboration_method = 2
args.framework = 'collaboration'
args.model_name = 't5-small'
args.losses = "sequential"
args.type_small = 'base'
args.arch_config = "scripts/configs/{}_conf.json".format(dataset)
args.distributed = False
args.multiGPU = True
args.fp16 = True
args.train = dataset
args.valid = dataset
args.test = dataset
# args.batch_size = 1
args.optim = 'adamw'
args.warmup_ratio = 0.05
args.lr = 1e-3
args.num_workers = 4
args.clip_grad_norm = 1.0
args.epoch = 10
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
#-----------------------------------------------------------


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

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

gpu = gpu0 # Change GPU ID
args.gpu = gpu
args.rank = gpu
print(f'Process Launching at GPU {gpu}')

torch.cuda.set_device('cuda:{}'.format(gpu))

comments = []
dsets = []
# if 'toys' in args.train:
#     dsets.append('toys')
# if 'beauty' in args.train:
#     dsets.append('beauty')
# if 'sports' in args.train:
#     dsets.append('sports')
dsets.append(dataset)
comments.append(''.join(dsets))
if args.backbone:
    comments.append(args.backbone)
comments.append(''.join(args.losses.split(',')))
if args.comment != '':
    comments.append(args.comment)
comment = '_'.join(comments)

if args.local_rank in [0, -1]:
    print(args)


def create_config(args):
    from transformers import T5Config, BartConfig

    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    print("args.backbone ", args.backbone)
    config = config_class.from_pretrained(args.backbone)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from src.tokenization import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.backbone

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )

    print(tokenizer_class, tokenizer_name)

    return tokenizer


def create_model(model_class, config=None):
    print(f'Building Model at GPU {args.gpu}')

    model_name = args.backbone

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model


config = create_config(args)

if args.tokenizer is None:
    args.tokenizer = args.backbone

tokenizer = create_tokenizer(args)

model_class = P5Pretraining
large_language_model = create_model(model_class, config)

large_language_model = large_language_model.cuda()

if 'p5' in args.tokenizer:
    large_language_model.resize_token_embeddings(tokenizer.vocab_size)

large_language_model.tokenizer = tokenizer

# args.load = "../snap/beauty-small.pth"
# args.load = "snap/beauty-small.pth"
# args.load = "beauty/BEST_EVAL_LOSS.pth"
# args.load = "{}/{}/BEST_EVAL_LOSS.pth".format(dataset, args.model_name)

# Load Checkpoint
from src.utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pprint

def load_checkpoint(ckpt_path):
    state_dict = load_state_dict(ckpt_path, 'cpu')
    results = large_language_model.load_state_dict(state_dict, strict=False)
    print('large_language_model loaded from ', ckpt_path)
    pprint(results)

# ckpt_path = args.load
# load_checkpoint(ckpt_path)

from src.all_amazon_templates import all_tasks as task_templates

# data_splits = load_pickle('../data/beauty/rating_splits_augmented.pkl')
data_splits = load_pickle('data/{}/rating_splits_augmented.pkl'.format(dataset))
test_review_data = data_splits['test']

print(len(test_review_data))

print(test_review_data[0])

# data_maps = load_json(os.path.join('../data', 'beauty', 'datamaps.json'))
data_maps = load_json(os.path.join('data', dataset, 'datamaps.json'))
print(len(data_maps['user2id'])) # number of users
print(len(data_maps['item2id'])) # number of items

# train_epoch = 10
args.losses = args.losses.split(",")
test_task_list_all = {
    'sequential': ['2-13'],  # or '2-3'
    'traditional': ['5-5']
}
# test_sample_numbers = {
#     'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)
# }
test_sample_numbers_all = {
    'sequential': (1, 1, 1), 'traditional': (1, 1)
}

debug = True
# debug = False

max_len = 50
large_pred_len = 50
# large_pred_len = 40
# large_pred_len = 100
large_pred_limit_len = 50
if dataset == "beauty":
    item_id_max = 12101
elif dataset == "sports":
    item_id_max = 18357
elif dataset == "toys":
    item_id_max = 11924
elif dataset == "yelp":
    item_id_max = 20033
else:
    raise NotImplementedError
item_id_min = 0

# rank_model = "small"
# rank_model = "both"
# rank_model = sys_args[7]
rank_model = "both"
args.dcc = True


for train_epoch_ in range(7, 8):
    train_epoch = train_epoch_ + 1
    for loss_ in args.losses:
        test_task_list = {loss_: test_task_list_all[loss_]}
        test_sample_numbers = {loss_: test_sample_numbers_all[loss_]}
        if loss_ is None:
            continue

        ### load large_model
        args.load_large = "checkpoint_large/{}/{}_{}/Epoch{}.pth".format(
            # dataset, args.model_name, loss_, str(train_epoch).zfill(2)
            dataset, args.model_name, loss_, str(10).zfill(2)
        )
        ckpt_path_large = args.load_large
        load_checkpoint(ckpt_path_large)

        ## load small_model
        # args.load_small = "checkpoint_collaboration_aug/{}/{}_{}_{}_{}/Epoch{}.pth".format(
        #     dataset, args.model_name, loss_, args.small_recommendation_model, args.type_small, str(train_epoch).zfill(2)
        # )
        args.load_small = "checkpoint_small_retrain/{}/{}_{}_{}/Epoch{}.pth".format(
            dataset, args.type_small, args.small_recommendation_model,loss_, str(train_epoch).zfill(2)
        )
        ckpt_path_small = args.load_small
        print("ckpt_path_large ", ckpt_path_large)
        print("ckpt_path_small ", ckpt_path_small)
        import sys

        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from util import consts
        import model
        from util import args_processing as ap

        # if args.framework == "small_recommendation_model_base":
        if args.type_small == "base":
            model_meta = model.get_model_meta(args.small_recommendation_model)  # type: model.ModelMeta
            model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

            # Construct model
            small_recommendation_model = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
            # # load_checkpoint(ckpt_path)
            # small_recommendation_model = torch.load(ckpt_path)
            state_dict = load_state_dict(ckpt_path_small, 'cpu')
            # results = model.load_state_dict(state_dict, strict=False)
            small_recommendation_model.load_state_dict(state_dict, strict=False)

        elif args.type_small == "duet":
            small_recommendation_model = torch.load(ckpt_path_small)

        else:
            raise NotImplementedError

        if debug:
            # args.framework = "small_recommendation_model_base"
            zeroshot_test_loader = get_loader(
                args,
                test_task_list,
                test_sample_numbers,
                split=args.test,
                mode='test',
                batch_size=args.batch_size,
                workers=args.num_workers,
                distributed=args.distributed
            )
            print(len(zeroshot_test_loader))

            all_info = []
            all_info_small = []
            from util import env
            device = env.get_device()
            small_recommendation_model.to(device)
            print("device ", device)
            pred_list_list = []
            small_model_top_k = 20
            pred_list = []
            y_list = []
            buffer = []
            all_score = []
            repeat = 0
            random.seed(2024)
            total = len(zeroshot_test_loader)
            choose = (total*request_ratio)//100
            p_list = [m for m in range(total)]
            random.shuffle(p_list)
            p_list = p_list[:choose]
            # llm_param = sum(p.numel() for p in large_language_model.parameters())
            # small_param = sum(p.numel() for p in small_recommendation_model.parameters())
            time_llm = []
            time_small = []
            time_all = []
            for i, batch in tqdm(enumerate(zeroshot_test_loader),total = len(zeroshot_test_loader)):
                REAL = False
                while True:
                    generated_sents_list_list = []
                    from collections import defaultdict
                    batch_small = defaultdict(list)
                    with torch.no_grad():
                        for key, value in batch.items():
                            try:
                                pass
                            except AttributeError:
                                pass
                        ### large model prediction
                        results = large_language_model.generate_step(batch)
                        # time_llm_begin = time.time()
                        beam_outputs = large_language_model.generate(
                            batch['input_ids'].to('cuda'),
                            max_length=max_len,
                            num_beams=large_pred_len,
                            no_repeat_ngram_size=0,
                            num_return_sequences=50,
                            early_stopping=True
                        )
                        # time_llm_end = time.time()
                        generated_sents = large_language_model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                        small_recommendation_index_list = []

                        cur_info = []
                        # candidates = [int(item) for item in batch[consts.FIELD_TARGET_ID].tolist()]  #
                        for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                            new_info = {}
                            new_info['target_item'] = item[1]
                            new_info['gen_item_list'] = generated_sents[j * large_pred_len: (j + 1) * large_pred_len]
                            generated_sents_list = []
                            for generated_item in new_info['gen_item_list']:
                                generated_item = generated_item.strip(".").strip(",").strip("-").strip(" ").strip("+")
                                try:
                                    if int(generated_item) > item_id_max or int(generated_item) < item_id_min:
                                        continue
                                    generated_sents_list.append(generated_item)  # str
                                except ValueError:
                                    continue
                            if rank_model == "both":
                                new_info['large_score'] = np.linspace(0.7, 0.3, large_pred_limit_len)
                            generated_sents_list = generated_sents_list[:large_pred_limit_len]
                            new_info['gen_item_list'] = generated_sents_list
                            for k in range(large_pred_limit_len - len(generated_sents_list)):
                                generated_sents_list.append("0")
                            generated_sents_list_list.append([int(k) for k in generated_sents_list])
                            cur_info.append(new_info)
                        generated_sents_list_array = np.array(generated_sents_list_list)
                        batch_small = batch
                        batch_lens = len(batch['target_text'])
                        batch_small[consts.FIELD_TARGET_ID] = torch.from_numpy(generated_sents_list_array)
                        batch_small[consts.FIELD_TARGET_ID] = batch_small[consts.FIELD_TARGET_ID].view(-1)
                        ### small model prediction
                        if not REAL:
                            for key, value in batch_small.items():
                                if key in {consts.FIELD_CLK_SEQUENCE, consts.FIELD_TRIGGER_SEQUENCE}:
                                        value = value.repeat_interleave(large_pred_limit_len, dim=0)
                                        batch_small[key] = value
                        # time_small_begin = time.time()
                        logits = small_recommendation_model({
                            key: value.to(device)
                            for key, value in batch_small.items()
                            if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE, consts.FIELD_TRIGGER_SEQUENCE}
                        })
                        # time_small_end = time.time()
                        # time_llm.append(time_llm_end-time_llm_begin)
                        # time_small.append(time_small_end-time_small_begin)
                        # time_all.append(time_small_end-time_llm_begin)

                        prob = torch.sigmoid(logits).view(batch_lens, large_pred_limit_len).detach().cpu().numpy()

                        target_id = torch.tensor(generated_sents_list_list).view(batch_lens,large_pred_limit_len)
                        prob_tensor = torch.from_numpy(prob)
                        sorted_indices = torch.argsort(prob_tensor, descending=True)
                        sorted_target = torch.gather(target_id, dim=1, index=sorted_indices).numpy()
                        sorted_target_id = [list(map(int, x)) for x in sorted_target]
                        score = 0.0
                        total = 0
                        total_0 = 0

                        for u in range(len(generated_sents_list_list)):
                            for index in range(len(generated_sents_list_list[u])):
                                item = sorted_target_id[u][index]
                                if item != 0:
                                    idx = generated_sents_list_list[u].index(item)
                                    score += abs(index-idx)
                                    total += 1

                        for u in range(len(generated_sents_list_list)):
                            for item in generated_sents_list_list[u]:
                                if item == 0:
                                    total_0+=1

                        score = score/total + pow(4+(total_0)*0.3,2)
                        if not REAL:
                            all_score.append(score)            
                        p = score_limit
                        
                        if request_mode == 'Random':
                            if not args.dcc or (i not in p_list) or REAL:
                                all_info += cur_info
                                for prob_ in prob:
                                    all_info_small.append(prob_)
                                pred_list_list.append(pred_list)
                                break
                            else:
                                batch['input_ids'] = batch['input_ids_real']
                                repeat += 1
                                REAL = True
                        else:
                            if not args.dcc or (score<p or REAL or repeat >= choose):
                                all_info += cur_info
                                for prob_ in prob:
                                    all_info_small.append(prob_)
                                pred_list_list.append(pred_list)
                                break
                            else:
                                batch['input_ids'] = batch['input_ids_real']
                                repeat += 1
                                REAL = True
                            

            gt = {}
            ui_scores = {}
            all_score.sort(reverse=True)
            length = len(all_score)
            with open(os.path.join(save_dir, "{}_{}_{}_{}.txt".format(dataset, args.model_name,args.small_recommendation_model, loss_)),
                          "a+") as writer:
                print("ratio: 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8",file = writer)
                print("score_limit:",all_score[int(length*0.05)],all_score[int(length*0.1)],all_score[int(length*0.2)],all_score[int(length*0.3)],all_score[int(length*0.4)],all_score[int(length*0.5)],all_score[int(length*0.6)],all_score[int(length*0.7)],all_score[int(length*0.8)],file=writer)
            
            metric_list = ["ndcg", "hit", "precision", "mrr"]
            from collections import defaultdict
            # results_dict = defaultdict(list)
            results_dict = {}
            gt = {}
            ui_scores_1 = {}
            ui_scores_2 = {}
            ui_scores_3 = {}
            ui_scores_4 = {}
            ui_scores_5 = {}
            for i, (info, info_small) in enumerate(zip(all_info, all_info_small)):
                gt[i] = [int(info['target_item'])]
                prob = info_small
                pred_dict_1 = {}
                pred_dict_2 = {}
                pred_dict_3 = {}
                pred_dict_4 = {}
                pred_dict_5 = {}
                candidate_len = large_pred_len
                for j in range(len(info['gen_item_list'])):
                    # try:
                        pred_dict_1[int(info['gen_item_list'][j])] = -(j + 1)

                        pred_dict_2[int(info['gen_item_list'][j])] = -(j + 1) * (1 / candidate_len) + 0.5
                        # pred_dict_2_pop[int(info['gen_item_list'][j])] = -(j + 1) * (1 / candidate_len) + 0.5
                        # index = info['candidate'].index(int(info['gen_item_list'][j]))
                        pr = prob[j] - 0.5
                        if pr > 0.4 or pr < -0.4:
                            pred_dict_2[int(info['gen_item_list'][j])] = pred_dict_2[int(info['gen_item_list'][j])] + pr

                        pred_dict_3[int(info['gen_item_list'][j])] = -(j + 1) * (1 / candidate_len) + pr
                        pred_dict_4[int(info['gen_item_list'][j])] = -(j + 1) * (1 / candidate_len) + pr * 2
                        pred_dict_5[int(info['gen_item_list'][j])] = prob[j]

                ui_scores_1[i] = pred_dict_1
                ui_scores_2[i] = pred_dict_2
                ui_scores_3[i] = pred_dict_3
                ui_scores_4[i] = pred_dict_4
                ui_scores_5[i] = pred_dict_5

            for index, ui_scores in enumerate([ui_scores_1, ui_scores_2, ui_scores_3, ui_scores_4, ui_scores_5,
                              ]):
                for num in [1, 5, 10, 20]:
                    evaluate_results_str, evaluate_results_dict = evaluate_all(ui_scores, gt, num)
                    for metric in metric_list:
                        decimal_num = 5
                        results_dict["{}{}".format(metric, num)] = "%.{}f".format(decimal_num)%evaluate_results_dict[metric]
                    # with open(os.path.join(save_dir, "{}_{}_{}.txt".format(dataset, args.large_model_name, loss_)), "a+") as writer:
                results_list = ["train_epoch={}, collaboration_method={}".format(train_epoch, index + 1)]
                for num in [1, 5, 10, 20]:
                    for metric in metric_list:
                        results_list.append("{}{}={}".format(metric, num, results_dict["{}{}".format(metric, num)]))

                with open(os.path.join(save_dir, "{}_{}_{}_{}.txt".format(dataset, args.model_name,
                                                                          args.small_recommendation_model, loss_)),
                          "a+") as writer:
                    print(", ".join(results_list), file=writer)