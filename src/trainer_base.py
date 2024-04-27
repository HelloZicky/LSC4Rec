import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pformat

proj_dir = Path(__file__).resolve().parent.parent

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


class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

    def create_config(self):
        from transformers import T5Config

        if 't5' in self.args.backbone:
            config_class = T5Config
        else:
            return None

        config = config_class.from_pretrained(self.args.backbone)

        args = self.args

        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout

        config.losses = args.losses

        return config

    def create_model(self, model_class, config=None, **kwargs):
        print(self.args)
        # if self.args.framework == "large_language_model":
        if self.args.framework == "large_language_model" or self.args.framework == "collaboration":
            print(f'Building Model at GPU {self.args.gpu}')

            model_name = self.args.backbone

            # model = model_class.from_pretrained(
            large_language_model = model_class.from_pretrained(
                model_name,
                config=config,
                **kwargs,
                # from_tf=True
            )

            # return model
            return large_language_model

        elif self.args.framework == "small_recommendation_model_base" or self.args.framework == "small_recommendation_model_duet":
            import model
            model_meta = model.get_model_meta(self.args.model)  # type: model.ModelMeta

            model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, self.args)  # type: dict
            small_recommendation_model = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module

            return small_recommendation_model

        # elif self.args.framework == "collaboration":
        #     print(f'Building Model at GPU {self.args.gpu}')
        #
        #     model_name = self.args.backbone
        #
        #     large_language_model = model_class.from_pretrained(
        #         model_name,
        #         config=config,
        #         **kwargs,
        #         # from_tf=True
        #     )
        #
        #     import model
        #     model_meta = model.get_model_meta(self.args.model)  # type: model.ModelMeta
        #
        #     model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict
        #     small_recommendation_model = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
        #
        #     return large_language_model, small_recommendation_model

        else:
            raise NotImplementedError

    def create_tokenizer(self, **kwargs):
        from transformers import T5Tokenizer, T5TokenizerFast
        from tokenization import P5Tokenizer, P5TokenizerFast

        if 'p5' in self.args.tokenizer:
            tokenizer_class = P5Tokenizer

        tokenizer_name = self.args.backbone

        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            max_length=self.args.max_text_length,
            do_lower_case=self.args.do_lower_case,
            **kwargs
            )

        return tokenizer

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        if 'adamw' in self.args.optim:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup

            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)
            
            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    # "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "params": [p for n, p in self.large_language_model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    # "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "params": [p for n, p in self.large_language_model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
            lr_scheduler = get_linear_schedule_with_warmup(
                optim, warmup_iters, t_total)

        else:
            optim = self.args.optimizer(
                list(self.large_language_model.parameters()), self.args.lr)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')
        results = self.large_language_model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.large_language_model.apply(init_bert_weights)
        self.large_language_model.init_weights()

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        if self.args.framework == "large_language_model":
            torch.save(self.large_language_model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))
        elif self.args.framework == "small_recommendation_model_base":
            torch.save(self.small_recommendation_model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))
        elif self.args.framework == "small_recommendation_model_duet":
            torch.save(self.small_recommendation_model.state_dict(), os.path.join(self.args.output, "%s_state_dict.pth" % name))
            torch.save(self.small_recommendation_model, os.path.join(self.args.output, "%s.pth" % name))
        elif self.args.framework == "collaboration":
            if not self.args.use_fusion_net:

                if self.args.retrain_type == "small":
                    if self.args.type_small == "base":
                        torch.save(self.small_recommendation_model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))
                        # torch.save(self.small_recommendation_model.state_dict(), os.path.join(
                        #     self.args.output + "_" + self.collaboration_method, "%s.pth" % name
                        # ))

                    elif self.args.type_small == "duet":
                        torch.save(self.small_recommendation_model.state_dict(), os.path.join(self.args.output, "%s_state_dict.pth" % name))
                        torch.save(self.small_recommendation_model, os.path.join(self.args.output, "%s.pth" % name))

                elif self.args.retrain_type == "large":
                    torch.save(self.large_language_model.state_dict(), os.path.join(self.args.output, "%s.pth" % name))

                else:
                    raise NotImplementedError

            elif self.args.use_fusion_net:
                torch.save(self.fusion_net.state_dict(), os.path.join(self.args.output, "%s.pth" % name))

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)
        results = self.large_language_model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)
