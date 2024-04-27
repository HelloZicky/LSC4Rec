import collections
import os
import random
from pathlib import Path
import logging
import shutil
from packaging import version

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from param import parse_args
from pretrain_data import get_loader
from utils import LossMeter
from dist_utils import reduce_dict

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

from trainer_base import TrainerBase


# The Trainer inherits TrainerBase in trainer_base.py
class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)
        self.args = args
        print(self.args.framework)
        if self.args.framework == "large_language_model" or self.args.framework == "collaboration":
            assert args.whole_word_embed
            from pretrain_model import P5Pretraining

            model_kwargs = {}
            model_class = P5Pretraining

            config = self.create_config()
            self.tokenizer = self.create_tokenizer()
            # if args.framework == "collaboration":
            #     self.large_language_model, self.small_recommendation_model = self.create_model(model_class, config, **model_kwargs)
            # else:
            #     # self.model = self.create_model(model_class, config, **model_kwargs)
            #     self.large_language_model = self.create_model(model_class, config, **model_kwargs)
            self.large_language_model = self.create_model(model_class, config, **model_kwargs)

            if 'p5' in self.args.tokenizer:
                self.large_language_model.resize_token_embeddings(self.tokenizer.vocab_size)

            self.large_language_model.tokenizer = self.tokenizer

            # Load Checkpoint
            self.start_epoch = None
            # self.collaboration_method = args.collaboration_method
            if args.load is not None:
                # ckpt_path = args.load + '.pth'
                ckpt_path = os.path.join(args.load, 'Epoch10.pth')
                self.load_checkpoint(ckpt_path)
                # self.start_epoch = int(args.load.split('Epoch-')[-1])

            # if args.load_small is not None:
            #     ckpt_path_small = args.load_small
            #     if args.retrain_type == "small" and args.collaboration_method == "small":
            #         from src.utils import load_state_dict
            #         # self.small_recommendation_model = torch.load(ckpt_path_small)
            #         state_dict = load_state_dict(ckpt_path_small, 'cpu')
            #         # results = model.load_state_dict(state_dict, strict=False)
            #         self.small_recommendation_model.load_state_dict(state_dict, strict=False)
            #     print("ckpt_path_small ", ckpt_path_small)

            if self.args.from_scratch:
                self.init_weights()

            # GPU Options
            print(f'large_language_model Launching at GPU {self.args.gpu}')
            if self.verbose:
                from time import time
                start = time()
            self.large_language_model = self.large_language_model.to(args.gpu)

            # Optimizer
            if train:
                self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

                if self.args.fp16 and _use_native_amp:
                    self.scaler = torch.cuda.amp.GradScaler()
                elif _use_apex:
                    self.large_language_model, self.optim = amp.initialize(
                        self.large_language_model, self.optim, opt_level='O1', verbosity=self.verbose)

            if args.multiGPU:
                if args.distributed:
                    self.large_language_model = DDP(self.large_language_model, device_ids=[args.gpu],
                                                    find_unused_parameters=True
                                                    )
            if self.verbose:
                print(f'It took {time() - start:.1f}s')

        elif self.args.framework == "small_recommendation_model_base" or self.args.framework == "small_recommendation_model_duet":
            print(args)
            print("hello")

        # elif self.args.framework == "collaboration":
        #     if not args.fusion_net:
        #         if args.retrain_large_model:
        #             ### load large_model
        #             assert args.whole_word_embed
        #             from pretrain_model import P5Pretraining
        #
        #             model_kwargs = {}
        #             model_class = P5Pretraining
        #
        #             config = self.create_config()
        #             self.tokenizer = self.create_tokenizer()
        #             # if args.framework == "collaboration":
        #             #     self.large_language_model, self.small_recommendation_model = self.create_model(model_class, config, **model_kwargs)
        #             # else:
        #             #     # self.model = self.create_model(model_class, config, **model_kwargs)
        #             #     self.large_language_model = self.create_model(model_class, config, **model_kwargs)
        #             self.large_language_model = self.create_model(model_class, config, **model_kwargs)
        #
        #             if 'p5' in self.args.tokenizer:
        #                 self.large_language_model.resize_token_embeddings(self.tokenizer.vocab_size)
        #
        #             self.large_language_model.tokenizer = self.tokenizer
        #
        #             # Load Checkpoint
        #             self.start_epoch = None
        #             if args.load is not None:
        #                 ckpt_path = args.load + '.pth'
        #                 self.load_checkpoint(ckpt_path)
        #                 self.start_epoch = int(args.load.split('Epoch-')[-1])
        #
        #             if self.args.from_scratch:
        #                 self.init_weights()
        #
        #             # GPU Options
        #             print(f'large_language_model Launching at GPU {self.args.gpu}')
        #             if self.verbose:
        #                 from time import time
        #                 start = time()
        #             self.large_language_model = self.large_language_model.to(args.gpu)
        #
        #             # Optimizer
        #             if train:
        #                 self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()
        #
        #                 if self.args.fp16 and _use_native_amp:
        #                     self.scaler = torch.cuda.amp.GradScaler()
        #                 elif _use_apex:
        #                     self.large_language_model, self.optim = amp.initialize(
        #                         self.large_language_model, self.optim, opt_level='O1', verbosity=self.verbose)
        #
        #             if args.multiGPU:
        #                 if args.distributed:
        #                     self.large_language_model = DDP(self.large_language_model, device_ids=[args.gpu],
        #                                                     find_unused_parameters=True
        #                                                     )
        #             if self.verbose:
        #                 print(f'It took {time() - start:.1f}s')
        #         if args.retrain_small_model:
        #             pass
        #     elif args.fusion_net:
        #         pass
        else:
            raise NotImplementedError

    def train(self):
        if args.framework == "large_language_model":
            LOSSES_NAME = self.args.LOSSES_NAME

            if self.args.dry:
                results = self.evaluate_epoch(epoch=0)

            if self.verbose:
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
                best_eval_loss = 100000.

                if 't5' in self.args.backbone:
                    project_name = "P5_Pretrain"

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)

            if self.args.distributed:
                dist.barrier()

            global_step = 0
            for epoch in range(self.args.epoch):
                if self.start_epoch is not None:
                    epoch += self.start_epoch
                if self.args.distributed:
                    self.train_loader.sampler.set_epoch(epoch)

                # Train
                self.large_language_model.train()

                if self.verbose:
                    pbar = tqdm(total=len(self.train_loader), ncols=275)

                epoch_results = {}
                for loss_name in LOSSES_NAME:
                    epoch_results[loss_name] = 0.
                    epoch_results[f'{loss_name}_count'] = 0

                for step_i, batch in enumerate(self.train_loader):
                    # if args.framework == "large_lanuage_model":
                    if self.args.fp16 and _use_native_amp:
                        with autocast():
                            if self.args.distributed:
                                results = self.large_language_model.module.train_step(batch)
                            else:
                                results = self.large_language_model.train_step(batch)
                    else:
                        if self.args.distributed:
                            results = self.large_language_model.module.train_step(batch)
                        else:
                            results = self.large_language_model.train_step(batch)

                    loss = results['loss']

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.scale(loss).backward()
                    elif self.args.fp16 and _use_apex:
                        with amp.scale_loss(loss, self.optim) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    loss = loss.detach()

                    # Update Parameters
                    if self.args.clip_grad_norm > 0:
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(self.large_language_model.parameters(),
                                                           self.args.clip_grad_norm)
                        elif self.args.fp16 and _use_apex:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.large_language_model.parameters(),
                                                           self.args.clip_grad_norm)

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                    # self.large_language_model.zero_grad()
                    for param in self.large_language_model.parameters():
                        param.grad = None

                    global_step += 1

                    if self.lr_scheduler:
                        if version.parse(torch.__version__) >= version.parse("1.4"):
                            lr = self.lr_scheduler.get_last_lr()[0]
                        else:
                            lr = self.lr_scheduler.get_lr()[0]
                    else:
                        try:
                            lr = self.optim.get_lr()[0]
                        except AttributeError:
                            lr = self.args.lr

                    for k, v in results.items():
                        if k in epoch_results:
                            if isinstance(v, int):
                                epoch_results[k] += v
                            elif isinstance(v, torch.Tensor):
                                epoch_results[k] += v.item()

                    if self.verbose and step_i % 200:
                        desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                        for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                            if loss_name in results:
                                loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                            if len(loss_meter) > 0:
                                loss_count = epoch_results[f'{loss_name}_count']
                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                        pbar.set_description(desc_str)
                        pbar.update(1)

                if self.verbose:
                    pbar.close()

                dist.barrier()

                results = reduce_dict(epoch_results, average=False)
                if self.verbose:
                    train_loss = results['total_loss']
                    train_loss_count = results['total_loss_count']

                    avg_train_loss = train_loss / train_loss_count
                    losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                    for name, loss in results.items():
                        if name[-4:] == 'loss':
                            loss_count = int(results[name + '_count'])
                            if loss_count > 0:
                                avg_loss = loss / loss_count
                                losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    losses_str += '\n'
                    print(losses_str)

                dist.barrier()

                if epoch > 10:
                    # Validation
                    valid_results = self.evaluate_epoch(epoch=epoch)

                    valid_results = reduce_dict(valid_results, average=False)
                    if self.verbose and step_i % 200:
                        valid_loss = valid_results['total_loss']
                        valid_loss_count = valid_results['total_loss_count']

                        avg_valid_loss = valid_loss / valid_loss_count
                        losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                        for name, loss in valid_results.items():
                            if name[-4:] == 'loss':
                                loss_count = int(valid_results[name + '_count'])
                                if loss_count > 0:
                                    avg_loss = loss / loss_count
                                    losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                        losses_str += '\n'
                        print(losses_str)

                    dist.barrier()

                    if self.verbose:
                        # Save
                        if avg_valid_loss < best_eval_loss:
                            best_eval_loss = avg_valid_loss
                            self.save("BEST_EVAL_LOSS")
                        self.save("Epoch%02d" % (epoch + 1))

                    dist.barrier()
                else:
                    # Skip validation
                    print("Skip validation for Epoch%02d" % (epoch + 1))
                    self.save("Epoch%02d" % (epoch + 1))

                    dist.barrier()

        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet":
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from util import consts
            import model
            from util import args_processing as ap

            if args.framework == "small_recommendation_model_base":
                model_meta = model.get_model_meta(self.args.small_recommendation_model)  # type: model.ModelMeta
            elif args.framework == "small_recommendation_model_duet":
                model_meta = model.get_model_meta("meta_" + self.args.small_recommendation_model)
            # model_meta = model.get_model_meta(self.args.small_recommendation_model)  # type: model.ModelMeta
            model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, self.args)  # type: dict

            # Construct model
            self.small_recommendation_model = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module

            # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict
            LOSSES_NAME = self.args.LOSSES_NAME

            if self.args.dry:
                results = self.evaluate_epoch(epoch=0)

            if self.verbose:
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
                best_eval_loss = 100000.

                if 't5' in self.args.backbone:
                    project_name = "P5_Pretrain"

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)

            if self.args.distributed:
                dist.barrier()

            global_step = 0
            optimizer = torch.optim.Adam(
                self.small_recommendation_model.parameters(),
                # lr=float(args.learning_rate)
                lr=float(args.lr)
            )
            # self.small_recommendation_model.train()
            from util import env
            device = env.get_device()
            print("device ", device)
            self.small_recommendation_model.to(device)
            # self.start_epoch = None
            for epoch in range(self.args.epoch):
                train_loss = 0
                train_loss_count = 0
                # train_loss_count = len(self.args.losses)
                # if self.start_epoch is not None:
                #     epoch += self.start_epoch
                if self.args.distributed:
                    self.train_loader.sampler.set_epoch(epoch)

                # Train
                # self.large_language_model.train()
                self.small_recommendation_model.train()

                if self.verbose:
                    pbar = tqdm(total=len(self.train_loader), ncols=275)

                epoch_results = {}
                for loss_name in LOSSES_NAME:
                    epoch_results[loss_name] = 0.
                    epoch_results[f'{loss_name}_count'] = 0

                for step_i, batch in enumerate(self.train_loader):
                    # for key, value in batch.items():
                    #     print("dddddddddd")
                    #     print(key)
                    #     print(value.size())
                    logits = self.small_recommendation_model({
                        key: value.to(device)
                        for key, value in batch.items()
                        # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                        if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE, consts.FIELD_TRIGGER_SEQUENCE}
                    })
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                    # for logit, label in zip(logits, batch[consts.FIELD_LABEL].view(-1, 1)):
                    #     print(logit, label, loss, sep="\t")

                    # train_loss += loss.detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                    train_loss += loss.clone().detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                    train_loss_count += len(batch[consts.FIELD_LABEL].view(-1, 1))
                    # pred, y = torch.sigmoid(logits), batch[consts.FIELD_LABEL].view(-1, 1)
                    # overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    global_step += 1

                    # if self.lr_scheduler:
                    #     if version.parse(torch.__version__) >= version.parse("1.4"):
                    #         lr = self.lr_scheduler.get_last_lr()[0]
                    #     else:
                    #         lr = self.lr_scheduler.get_lr()[0]
                    # else:
                    #     try:
                    #         lr = self.optim.get_lr()[0]
                    #     except AttributeError:
                    #         lr = self.args.lr
                    #
                    # for k, v in results.items():
                    #     if k in epoch_results:
                    #         if isinstance(v, int):
                    #             epoch_results[k] += v
                    #         elif isinstance(v, torch.Tensor):
                    #             epoch_results[k] += v.item()
                    lr = self.args.lr
                    # results = {}
                    if self.verbose and step_i % 200:
                        desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                        for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                            # if loss_name in results:
                            if loss_name in self.args.LOSSES_NAME:
                                # loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                                loss_meter.update(loss)
                            if len(loss_meter) > 0:
                                # loss_count = epoch_results[f'{loss_name}_count']
                                # desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                                loss_count = loss
                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                        pbar.set_description(desc_str)
                        pbar.update(1)

                if self.verbose:
                    pbar.close()

                dist.barrier()

                # results = reduce_dict(epoch_results, average=False)
                if self.verbose:
                    # train_loss = results['total_loss']
                    # train_loss_count = results['total_loss_count']
                    # train_loss_count =

                    avg_train_loss = train_loss / train_loss_count
                    # avg_train_loss = loss / train_loss_count
                    losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                    # for name, loss in results.items():
                    #     if name[-4:] == 'loss':
                    #         loss_count = int(results[name + '_count'])
                    #         if loss_count > 0:
                    #             avg_loss = loss / loss_count
                    #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                    losses_str += f"{self.args.losses}_loss ({train_loss_count}): {avg_train_loss:.3f} "

                    losses_str += '\n'
                    print(losses_str)

                dist.barrier()

                if epoch > 10:
                    # Validation
                    # valid_results = self.evaluate_epoch(epoch=epoch)
                    # valid_loss = self.evaluate_epoch(epoch=epoch)
                    valid_results = self.evaluate_epoch(epoch=epoch)
                    # valid_results = reduce_dict(valid_results, average=False)
                    if self.verbose and step_i % 200:
                        valid_loss = valid_results['total_loss']
                        valid_loss_count = valid_results['total_loss_count']
                        # valid_loss_count = len(self.args.losses)
                        avg_valid_loss = valid_loss / valid_loss_count
                        losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                        # for name, loss in valid_results.items():
                        #     if name[-4:] == 'loss':
                        #         loss_count = int(valid_results[name + '_count'])
                        #         if loss_count > 0:
                        #             avg_loss = loss / loss_count
                        #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                        # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                        losses_str += f"{self.args.losses}_loss ({valid_loss_count}): {avg_valid_loss:.3f} "

                        losses_str += '\n'
                        print(losses_str)

                    dist.barrier()

                    if self.verbose:
                        # Save
                        if avg_valid_loss < best_eval_loss:
                            best_eval_loss = avg_valid_loss
                            self.save("BEST_EVAL_LOSS")
                        self.save("Epoch%02d" % (epoch + 1))

                    dist.barrier()
                else:
                    # Skip validation
                    print("Skip validation for Epoch%02d" % (epoch + 1))
                    self.save("Epoch%02d" % (epoch + 1))

                    dist.barrier()

        elif args.framework == "collaboration":
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
            from util import consts
            import model
            from util import args_processing as ap
            args.use_fusion_net = (args.fusion_net is not None)
            # if args.retrain_large_model
            if args.train == "beauty" and args.valid == "beauty":
                item_id_max = 12101
            elif args.train == "sports" and args.valid == "sports":
                item_id_max = 18357
            elif args.train == "toys" and args.valid == "toys":
                item_id_max = 11924
            elif args.train == "yelp" and args.valid == "yelp":
                item_id_max = 20033
            else:
                raise NotImplementedError
            item_id_min = 0
            if not args.use_fusion_net:
                if args.retrain_type == "large":
                    # if args.framework == "small_recommendation_model_base":
                    if args.type_small == "base":
                        small_model_name = self.args.small_recommendation_model
                    # elif args.framework == "small_recommendation_model_duet":
                    elif args.type_small == "duet":
                        small_model_name = "meta_" + self.args.small_recommendation_model
                    else:
                        raise NotImplementedError

                elif args.retrain_type == "small":
                    # if args.framework == "small_recommendation_model_base":
                    if args.type_small == "base":
                        if self.args.collaboration_method == 1:
                            small_model_name = self.args.small_recommendation_model + "_fusion"
                        elif self.args.collaboration_method == 2:
                            small_model_name = self.args.small_recommendation_model
                    # elif args.framework == "small_recommendation_model_duet":
                    elif args.type_small == "duet":
                        small_model_name = "meta_" + self.args.small_recommendation_model + "_fusion"
                    else:
                        raise NotImplementedError

                else:
                    raise NotImplementedError

            elif args.use_fusion_net:
                if args.type_small == "base":
                    small_model_name = self.args.small_recommendation_model
                # elif args.framework == "small_recommendation_model_duet":
                elif args.type_small == "duet":
                    small_model_name = "meta_" + self.args.small_recommendation_model
                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError

            model_meta_small = model.get_model_meta(small_model_name)  # type: model.ModelMeta
            # model_meta = model.get_model_meta(self.args.small_recommendation_model)  # type: model.ModelMeta
            model_conf_small, raw_model_conf_small = ap.parse_arch_config_from_args(model_meta_small, self.args)  # type: dict

            # Construct model
            self.small_recommendation_model = model_meta_small.model_builder(model_conf=model_conf_small)  # type: torch.nn.module

            ### large model
            LOSSES_NAME = self.args.LOSSES_NAME

            if self.args.dry:
                results = self.evaluate_epoch(epoch=0)

            if self.verbose:
                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
                best_eval_loss = 100000.

                if 't5' in self.args.backbone:
                    project_name = "P5_Pretrain"

                src_dir = Path(__file__).resolve().parent
                base_path = str(src_dir.parent)
                src_dir = str(src_dir)

            if self.args.distributed:
                dist.barrier()

            if args.load is not None:
                # ckpt_path = args.load + '.pth'
                ckpt_path = os.path.join(args.load, 'Epoch10.pth')
                self.load_checkpoint(ckpt_path)
                print("ckpt_path_large ", ckpt_path)

            if args.load_small is not None:
                ckpt_path_small = os.path.join(args.load_small, 'Epoch10.pth')
                # if args.retrain_type == "small" and args.collaboration_method == 2:
                if args.retrain_type == "small":
                    from src.utils import load_state_dict
                    # self.small_recommendation_model = torch.load(ckpt_path_small)
                    state_dict = load_state_dict(ckpt_path_small, 'cpu')
                    # results = model.load_state_dict(state_dict, strict=False)
                    self.small_recommendation_model.load_state_dict(state_dict, strict=False)
                print("ckpt_path_small ", ckpt_path_small)

            global_step = 0
            for epoch in range(self.args.epoch):
                if self.start_epoch is not None:
                    epoch += self.start_epoch
                if self.args.distributed:
                    self.train_loader.sampler.set_epoch(epoch)

                # Train
                self.large_language_model.train()

                if self.verbose:
                    pbar = tqdm(total=len(self.train_loader), ncols=275)

                epoch_results = {}
                for loss_name in LOSSES_NAME:
                    epoch_results[loss_name] = 0.
                    epoch_results[f'{loss_name}_count'] = 0

                if not args.use_fusion_net:
                    # if args.retrain_large_model and (not args.retrain_small_model):
                    if args.retrain_type == "large":
                        # with torch.no_grad():
                        for step_i, batch in enumerate(self.train_loader):
                            # if args.framework == "large_language_model":
                            if self.args.fp16 and _use_native_amp:
                                with autocast():
                                    if self.args.distributed:
                                        results = self.large_language_model.module.train_step(batch)
                                    else:
                                        results = self.large_language_model.train_step(batch)
                            else:
                                if self.args.distributed:
                                    results = self.large_language_model.module.train_step(batch)
                                else:
                                    results = self.large_language_model.train_step(batch)

                            loss = results['loss']

                            # if self.args.fp16 and _use_native_amp:
                            #     self.scaler.scale(loss).backward()
                            # elif self.args.fp16 and _use_apex:
                            #     with amp.scale_loss(loss, self.optim) as scaled_loss:
                            #         scaled_loss.backward()
                            # else:
                            #     loss.backward()
                            #
                            # loss = loss.detach()
                            #
                            # # Update Parameters
                            # if self.args.clip_grad_norm > 0:
                            #     if self.args.fp16 and _use_native_amp:
                            #         self.scaler.unscale_(self.optim)
                            #         torch.nn.utils.clip_grad_norm_(self.large_language_model.parameters(),
                            #                                        self.args.clip_grad_norm)
                            #     elif self.args.fp16 and _use_apex:
                            #         torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.args.clip_grad_norm)
                            #     else:
                            #         torch.nn.utils.clip_grad_norm_(self.large_language_model.parameters(),
                            #                                        self.args.clip_grad_norm)
                            #
                            # if self.args.fp16 and _use_native_amp:
                            #     self.scaler.step(self.optim)
                            #     self.scaler.update()
                            # else:
                            #     self.optim.step()
                            #
                            # if self.lr_scheduler:
                            #     self.lr_scheduler.step()

                            # # self.large_language_model.zero_grad()
                            # for param in self.large_language_model.parameters():
                            #     param.grad = None

                            global_step += 1

                            if self.lr_scheduler:
                                if version.parse(torch.__version__) >= version.parse("1.4"):
                                    lr = self.lr_scheduler.get_last_lr()[0]
                                else:
                                    lr = self.lr_scheduler.get_lr()[0]
                            else:
                                try:
                                    lr = self.optim.get_lr()[0]
                                except AttributeError:
                                    lr = self.args.lr

                            for k, v in results.items():
                                if k in epoch_results:
                                    if isinstance(v, int):
                                        epoch_results[k] += v
                                    elif isinstance(v, torch.Tensor):
                                        epoch_results[k] += v.item()

                            if self.verbose and step_i % 200:
                                desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                                for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                                    if loss_name in results:
                                        loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                                    if len(loss_meter) > 0:
                                        loss_count = epoch_results[f'{loss_name}_count']
                                        desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                                pbar.set_description(desc_str)
                                pbar.update(1)

                        if self.verbose:
                            pbar.close()

                        dist.barrier()

                        # results = reduce_dict(epoch_results, average=False)
                        if self.verbose:
                            # train_loss = results['total_loss']
                            # train_loss_count = results['total_loss_count']
                            # train_loss_count =

                            avg_train_loss = train_loss / train_loss_count
                            # avg_train_loss = loss / train_loss_count
                            losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                            # for name, loss in results.items():
                            #     if name[-4:] == 'loss':
                            #         loss_count = int(results[name + '_count'])
                            #         if loss_count > 0:
                            #             avg_loss = loss / loss_count
                            #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                            # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                            losses_str += f"{self.args.losses}_loss ({train_loss_count}): {avg_train_loss:.3f} "

                            losses_str += '\n'
                            print(losses_str)

                        dist.barrier()

                        if epoch > 10:
                            # Validation
                            # valid_results = self.evaluate_epoch(epoch=epoch)
                            # valid_loss = self.evaluate_epoch(epoch=epoch)
                            valid_results = self.evaluate_epoch(epoch=epoch)
                            # valid_results = reduce_dict(valid_results, average=False)
                            if self.verbose and step_i % 200:
                                valid_loss = valid_results['total_loss']
                                valid_loss_count = valid_results['total_loss_count']
                                # valid_loss_count = len(self.args.losses)
                                avg_valid_loss = valid_loss / valid_loss_count
                                losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                                # for name, loss in valid_results.items():
                                #     if name[-4:] == 'loss':
                                #         loss_count = int(valid_results[name + '_count'])
                                #         if loss_count > 0:
                                #             avg_loss = loss / loss_count
                                #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                losses_str += f"{self.args.losses}_loss ({valid_loss_count}): {avg_valid_loss:.3f} "

                                losses_str += '\n'
                                print(losses_str)

                            dist.barrier()

                            if self.verbose:
                                # Save
                                if avg_valid_loss < best_eval_loss:
                                    best_eval_loss = avg_valid_loss
                                    self.save("BEST_EVAL_LOSS")
                                self.save("Epoch%02d" % (epoch + 1))

                            dist.barrier()
                        else:
                            # Skip validation
                            print("Skip validation for Epoch%02d" % (epoch + 1))
                            self.save("Epoch%02d" % (epoch + 1))

                            dist.barrier()


                    # elif args.retrain_small_model and (not args.retrain_large_model):
                    elif args.retrain_type == "small":
                        # collaboration_method = 2
                        # collaboration_method = 1
                        # collaboration_method = self.collaboration_method
                        collaboration_method = args.collaboration_method
                        if collaboration_method == 1:
                            # if args.framework == "small_recommendation_model_base":
                            #     model_meta = model.get_model_meta(
                            #         self.args.small_recommendation_model)  # type: model.ModelMeta
                            # elif args.framework == "small_recommendation_model_duet":
                            #     model_meta = model.get_model_meta("meta_" + self.args.small_recommendation_model)

                            # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict
                            LOSSES_NAME = self.args.LOSSES_NAME

                            if self.args.dry:
                                results = self.evaluate_epoch(epoch=0)

                            if self.verbose:
                                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
                                best_eval_loss = 100000.

                                if 't5' in self.args.backbone:
                                    project_name = "P5_Pretrain"

                                src_dir = Path(__file__).resolve().parent
                                base_path = str(src_dir.parent)
                                src_dir = str(src_dir)

                            if self.args.distributed:
                                dist.barrier()

                            global_step = 0
                            optimizer_small = torch.optim.Adam(
                                self.small_recommendation_model.parameters(),
                                # lr=float(args.learning_rate)
                                lr=float(args.lr)
                            )
                            # self.small_recommendation_model.train()
                            from util import env
                            device = env.get_device()
                            print("device ", device)
                            self.small_recommendation_model.to(device)
                            # self.start_epoch = None
                            for epoch in range(self.args.epoch):
                                train_loss = 0
                                train_loss_count = 0
                                # train_loss_count = len(self.args.losses)
                                # if self.start_epoch is not None:
                                #     epoch += self.start_epoch
                                if self.args.distributed:
                                    self.train_loader.sampler.set_epoch(epoch)

                                # Train
                                # self.large_language_model.train()
                                self.small_recommendation_model.train()

                                if self.verbose:
                                    pbar = tqdm(total=len(self.train_loader), ncols=275)

                                epoch_results = {}
                                for loss_name in LOSSES_NAME:
                                    epoch_results[loss_name] = 0.
                                    epoch_results[f'{loss_name}_count'] = 0

                                # for i, batch in tqdm(enumerate(self.train_loader)):
                                for step_i, batch in tqdm(enumerate(self.train_loader)):
                                    ### -------------large model---------------
                                    # with torch.no_grad():
                                    results = self.large_language_model.module.generate_step(batch)
                                    beam_outputs = self.large_language_model.module.generate(
                                        batch['input_ids'].to('cuda'),
                                        max_length=50,
                                        num_beams=30,
                                        no_repeat_ngram_size=0,
                                        num_return_sequences=30,
                                        early_stopping=True
                                    )
                                    generated_sents = self.large_language_model.module.tokenizer.batch_decode(
                                        beam_outputs, skip_special_tokens=True)
                                    # print(" +_+ " * 20)
                                    # print(generated_sents)
                                    generated_sents_list = []
                                    large_pred_list_len = 20
                                    for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                                        # new_info = {}
                                        # new_info['target_item'] = item[1]
                                        # new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
                                        # all_info.append(new_info)
                                        generated_sents_list_ = []
                                        for generated_item in generated_sents[j * 30: (j + 1) * 30]:
                                            # print("-=-=-")
                                            # print(generated_sents[j * 25: (j + 1) * 25])
                                            # print("[]")
                                            # print(generated_item)
                                            if len(generated_sents_list_) == large_pred_list_len:
                                                break
                                            try:
                                                generated_item = int(generated_item)
                                                # if generated_item > self.small_recommendation_model.id_vocab:
                                                if generated_item > item_id_max or generated_item < item_id_min:
                                                # if generated_item > model_conf_small.id_vocab - 1 or generated_item < 0:
                                                    continue
                                                generated_sents_list_.append(generated_item)
                                                # # print("!!!")
                                                # print(len(generated_sents_list_))
                                            except ValueError:
                                                continue
                                        # print("!!!")
                                        # print(len(generated_sents_list_))
                                        # print(generated_sents_list_)
                                        for i in range(large_pred_list_len - len(generated_sents_list_)):
                                            generated_sents_list_.append(0)
                                        generated_sents_list.append(generated_sents_list_)
                                    # print("???")
                                    # print(generated_sents_list)

                                    # generated_sents_feature = torch.from_numpy(np.array([int(i) for i in generated_sents]))
                                    generated_sents_feature = torch.from_numpy(np.array(generated_sents_list)).to(device)

                                    # for i in generated_sents:
                                    #     if i.split(".") == 1 and i.split(",") == 1:
                                    #         generated_sents_list.append(int(i))

                                    # print("=-" * 50)
                                    # for key, value in batch.items():
                                    #     print("-" * 50)
                                    #     print(key)
                                    #     print(value)
                                    #     # print(value.size())
                                    # print("-" * 50)
                                    # print("beam_outputs")
                                    # print(beam_outputs)
                                    # print(beam_outputs.size())
                                    # print("generated_sents")
                                    # print(np.array(generated_sents).shape)
                                    # print(generated_sents_feature.size())
                                    # print(generated_sents_feature)
                                    # print(generated_sents_feature[0])

                                    # generated_sents_feature = generated_sents_feature.view(self.args.batch_size, -1)
                                    # print(generated_sents_feature.size())
                                    # print(generated_sents_feature[0])

                                    ### -------------small model---------------
                                    logits = self.small_recommendation_model({
                                        key: value.to(device)
                                        for key, value in batch.items()
                                        # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                                        if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE,
                                                   consts.FIELD_TRIGGER_SEQUENCE}
                                    }, generated_sents_feature)
                                    criterion = nn.BCEWithLogitsLoss()
                                    loss = criterion(logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                                    # for logit, label in zip(logits, batch[consts.FIELD_LABEL].view(-1, 1)):
                                    #     print(logit, label, loss, sep="\t")

                                    # train_loss += loss.detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                                    train_loss += loss.clone().detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                                    train_loss_count += len(batch[consts.FIELD_LABEL].view(-1, 1))
                                    # pred, y = torch.sigmoid(logits), batch[consts.FIELD_LABEL].view(-1, 1)
                                    # overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
                                    optimizer_small.zero_grad()
                                    loss.backward()
                                    optimizer_small.step()

                                    global_step += 1

                                    lr = self.args.lr
                                    # results = {}
                                    if self.verbose and step_i % 200:
                                        desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                                        for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                                            # if loss_name in results:
                                            if loss_name in self.args.LOSSES_NAME:
                                                # loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                                                loss_meter.update(loss)
                                            if len(loss_meter) > 0:
                                                # loss_count = epoch_results[f'{loss_name}_count']
                                                # desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                                                loss_count = loss
                                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                                        pbar.set_description(desc_str)
                                        pbar.update(1)

                                if self.verbose:
                                    pbar.close()

                                dist.barrier()

                                # results = reduce_dict(epoch_results, average=False)
                                if self.verbose:
                                    # train_loss = results['total_loss']
                                    # train_loss_count = results['total_loss_count']
                                    # train_loss_count =

                                    avg_train_loss = train_loss / train_loss_count
                                    # avg_train_loss = loss / train_loss_count
                                    losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                                    # for name, loss in results.items():
                                    #     if name[-4:] == 'loss':
                                    #         loss_count = int(results[name + '_count'])
                                    #         if loss_count > 0:
                                    #             avg_loss = loss / loss_count
                                    #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                                    # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                    losses_str += f"{self.args.losses}_loss ({train_loss_count}): {avg_train_loss:.3f} "

                                    losses_str += '\n'
                                    print(losses_str)

                                dist.barrier()

                                if epoch > 10:
                                    # Validation
                                    # valid_results = self.evaluate_epoch(epoch=epoch)
                                    # valid_loss = self.evaluate_epoch(epoch=epoch)
                                    valid_results = self.evaluate_epoch(epoch=epoch)
                                    # valid_results = reduce_dict(valid_results, average=False)
                                    if self.verbose and step_i % 200:
                                        valid_loss = valid_results['total_loss']
                                        valid_loss_count = valid_results['total_loss_count']
                                        # valid_loss_count = len(self.args.losses)
                                        avg_valid_loss = valid_loss / valid_loss_count
                                        losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                                        # for name, loss in valid_results.items():
                                        #     if name[-4:] == 'loss':
                                        #         loss_count = int(valid_results[name + '_count'])
                                        #         if loss_count > 0:
                                        #             avg_loss = loss / loss_count
                                        #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                        # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                        losses_str += f"{self.args.losses}_loss ({valid_loss_count}): {avg_valid_loss:.3f} "

                                        losses_str += '\n'
                                        print(losses_str)

                                    dist.barrier()

                                    if self.verbose:
                                        # Save
                                        if avg_valid_loss < best_eval_loss:
                                            best_eval_loss = avg_valid_loss
                                            self.save("BEST_EVAL_LOSS")
                                        self.save("Epoch%02d" % (epoch + 1))

                                    dist.barrier()
                                else:
                                    # Skip validation
                                    print("Skip validation for Epoch%02d" % (epoch + 1))
                                    self.save("Epoch%02d" % (epoch + 1))

                                    dist.barrier()

                        elif collaboration_method == 2:
                            max_len = 50
                            large_pred_len = 20
                            # large_pred_len = 40
                            # large_pred_len = 100
                            large_pred_limit_len = 20
                            # if dataset == "beauty":
                            #     item_id_max = 12101
                            # elif dataset == "sports":
                            #     item_id_max = 18357
                            # elif dataset == "toys":
                            #     item_id_max = 11924
                            # elif dataset == "yelp":
                            #     item_id_max = 20033
                            # else:
                            #     raise NotImplementedError
                            # item_id_min = 0

                            LOSSES_NAME = self.args.LOSSES_NAME

                            if self.args.dry:
                                results = self.evaluate_epoch(epoch=0)

                            if self.verbose:
                                loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]
                                best_eval_loss = 100000.

                                if 't5' in self.args.backbone:
                                    project_name = "P5_Pretrain"

                                src_dir = Path(__file__).resolve().parent
                                base_path = str(src_dir.parent)
                                src_dir = str(src_dir)

                            if self.args.distributed:
                                dist.barrier()

                            global_step = 0
                            optimizer_small = torch.optim.Adam(
                                self.small_recommendation_model.parameters(),
                                # lr=float(args.learning_rate)
                                lr=float(args.lr)
                            )
                            # self.small_recommendation_model.train()
                            from util import env
                            device = env.get_device()
                            print("device ", device)
                            self.small_recommendation_model.to(device)
                            # self.start_epoch = None
                            for epoch in range(self.args.epoch):
                                train_loss = 0
                                train_loss_count = 0
                                # train_loss_count = len(self.args.losses)
                                # if self.start_epoch is not None:
                                #     epoch += self.start_epoch
                                if self.args.distributed:
                                    self.train_loader.sampler.set_epoch(epoch)

                                # Train
                                # self.large_language_model.train()
                                self.small_recommendation_model.train()

                                if self.verbose:
                                    pbar = tqdm(total=len(self.train_loader), ncols=275)

                                epoch_results = {}
                                for loss_name in LOSSES_NAME:
                                    epoch_results[loss_name] = 0.
                                    epoch_results[f'{loss_name}_count'] = 0

                                # for i, batch in tqdm(enumerate(self.train_loader)):
                                for step_i, batch in tqdm(enumerate(self.train_loader)):
                                    generated_sents_list_list = []
                                    label_list_list = []
                                    from collections import defaultdict
                                    batch_small = defaultdict(list)
                                    ### -------------large model---------------
                                    # with torch.no_grad():
                                    results = self.large_language_model.module.generate_step(batch)
                                    # beam_outputs = large_language_model.generate(
                                    beam_outputs = self.large_language_model.module.generate(
                                        batch['input_ids'].to('cuda'),
                                        max_length=max_len,
                                        num_beams=large_pred_len,
                                        no_repeat_ngram_size=0,
                                        num_return_sequences=20,
                                        early_stopping=True
                                    )
                                    # generated_sents = large_language_model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                                    generated_sents = self.large_language_model.module.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                                    small_recommendation_index_list = []
                                    for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                                        new_info = {}
                                        new_info['target_item'] = item[1]
                                        new_info['gen_item_list'] = generated_sents[j * large_pred_len: (j + 1) * large_pred_len]
                                        generated_sents_list = []
                                        label_list = []
                                        for generated_item in new_info['gen_item_list']:
                                            generated_item = generated_item.strip(".").strip(",").strip("-").strip(" ")
                                            try:
                                                if int(generated_item) > item_id_max or int(generated_item) < item_id_min:
                                                    continue
                                                generated_sents_list.append(generated_item)  # str
                                            except ValueError:
                                                continue
                                        #数据增强处理
                                        first,second,third = ['0'],[],[]
                                        for item in generated_sents_list:
                                            if len(item) <= 2:
                                                third.append(item)
                                            elif len(item) <= 4:
                                                second.append(item[:-2])
                                                third.append(item[-2:])
                                            else:
                                                first.append(item[:-4])
                                                second.append(item[-4:-2])
                                                third.append(item[-2:])
                                        first,second,third = set(first),set(second),set(third)
                                        total = [x+y+z for x in first for y in second for z in third]
                                        total = [str(int(x)) for x in total]
                                        random.shuffle(total)
                                        total = total[:79] + ['0'] * (79-len(total))
                                        for item in total:
                                            generated_sents_list.append(item)
                                        #end数据增强处理

                                        label_list.append(1.0)
                                        for index, candidate in enumerate(generated_sents_list):
                                            if candidate == str(batch[consts.FIELD_TARGET_ID].numpy().tolist()[j]):
                                                label_list.append(1.0)
                                            elif candidate != str(batch[consts.FIELD_TARGET_ID].numpy().tolist()[j]):
                                                label_list.append(0.0)
                                            generated_sents_list[index] = int(generated_sents_list[index])

                                        generated_sents_list = (
                                            [int(batch[consts.FIELD_TARGET_ID].numpy().tolist()[j])] +
                                            generated_sents_list[:99]# generated_sents_list[:large_pred_limit_len]
                                        )

                                        # new_info['gen_item_list'] = generated_sents_list[:large_pred_limit_len]
                                        new_info['gen_item_list'] = generated_sents_list
                                        for i in range(100 - len(generated_sents_list)):
                                            generated_sents_list.append(0)
                                            label_list.append(0)
                                        generated_sents_list_list.append(generated_sents_list)
                                        label_list_list.append(label_list)
                                    # print('============================')
                                    # print(generated_sents_list_list)
                                    # print('==============================')
                                    generated_sents_list_array = np.array(generated_sents_list_list)
                                    label_list_array = np.array(label_list_list)
                                    # print("=" * 50)
                                    # print(batch_small)
                                    # print(batch_small.items())
                                    batch_small = batch

                                    batch_small[consts.FIELD_TARGET_ID] = torch.from_numpy(generated_sents_list_array)
                                    batch_small[consts.FIELD_LABEL] = torch.from_numpy(label_list_array)
                                    batch_small[consts.FIELD_TARGET_ID] = batch_small[consts.FIELD_TARGET_ID].view(-1)
                                    batch_small[consts.FIELD_LABEL] = batch_small[consts.FIELD_LABEL].view(-1)
                                    ### small model prediction
                                    for key, value in batch_small.items():
                                        # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                                        # if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE, consts.FIELD_TRIGGER_SEQUENCE}:
                                        if key in {consts.FIELD_CLK_SEQUENCE, consts.FIELD_TRIGGER_SEQUENCE}:
                                            # print("=" * 50)
                                            # print(key)
                                            # print(value.size())
                                            # value = value.repeat_interleave(large_pred_limit_len, dim=0)
                                            value = value.repeat_interleave(100, dim=0)
                                            # print(value.size())
                                            batch_small[key] = value

                                    logits = self.small_recommendation_model({
                                        # key: value.repeat_interleave(large_pred_limit_len, dim=0).to(device)
                                        key: value.to(device)
                                        # for key, value in batch.items()
                                        for key, value in batch_small.items()
                                        # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                                        if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE,
                                                   consts.FIELD_TRIGGER_SEQUENCE}
                                    })

                                    criterion = nn.BCEWithLogitsLoss()
                                    loss = criterion(logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                                    # for logit, label in zip(logits, batch[consts.FIELD_LABEL].view(-1, 1)):
                                    #     print(logit, label, loss, sep="\t")

                                    # train_loss += loss.detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                                    train_loss += loss.clone().detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                                    train_loss_count += len(batch[consts.FIELD_LABEL].view(-1, 1))
                                    # pred, y = torch.sigmoid(logits), batch[consts.FIELD_LABEL].view(-1, 1)
                                    # overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
                                    optimizer_small.zero_grad()
                                    loss.backward()
                                    optimizer_small.step()

                                    global_step += 1

                                    lr = self.args.lr
                                    # results = {}
                                    if self.verbose and step_i % 200:
                                        desc_str = f'Epoch {epoch} | LR {lr:.6f} |'

                                        for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                                            # if loss_name in results:
                                            if loss_name in self.args.LOSSES_NAME:
                                                # loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                                                loss_meter.update(loss)
                                            if len(loss_meter) > 0:
                                                # loss_count = epoch_results[f'{loss_name}_count']
                                                # desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                                                loss_count = loss
                                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                                        pbar.set_description(desc_str)
                                        pbar.update(1)

                                if self.verbose:
                                    pbar.close()

                                dist.barrier()

                                # results = reduce_dict(epoch_results, average=False)
                                if self.verbose:
                                    # train_loss = results['total_loss']
                                    # train_loss_count = results['total_loss_count']
                                    # train_loss_count =

                                    avg_train_loss = train_loss / train_loss_count
                                    # avg_train_loss = loss / train_loss_count
                                    losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                                    # for name, loss in results.items():
                                    #     if name[-4:] == 'loss':
                                    #         loss_count = int(results[name + '_count'])
                                    #         if loss_count > 0:
                                    #             avg_loss = loss / loss_count
                                    #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                                    # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                    losses_str += f"{self.args.losses}_loss ({train_loss_count}): {avg_train_loss:.3f} "

                                    losses_str += '\n'
                                    print(losses_str)

                                dist.barrier()

                                if epoch > 10:
                                    # Validation
                                    # valid_results = self.evaluate_epoch(epoch=epoch)
                                    # valid_loss = self.evaluate_epoch(epoch=epoch)
                                    valid_results = self.evaluate_epoch(epoch=epoch)
                                    # valid_results = reduce_dict(valid_results, average=False)
                                    if self.verbose and step_i % 200:
                                        valid_loss = valid_results['total_loss']
                                        valid_loss_count = valid_results['total_loss_count']
                                        # valid_loss_count = len(self.args.losses)
                                        avg_valid_loss = valid_loss / valid_loss_count
                                        losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                                        # for name, loss in valid_results.items():
                                        #     if name[-4:] == 'loss':
                                        #         loss_count = int(valid_results[name + '_count'])
                                        #         if loss_count > 0:
                                        #             avg_loss = loss / loss_count
                                        #             losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                        # losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "
                                        losses_str += f"{self.args.losses}_loss ({valid_loss_count}): {avg_valid_loss:.3f} "

                                        losses_str += '\n'
                                        print(losses_str)

                                    dist.barrier()

                                    if self.verbose:
                                        # Save
                                        if avg_valid_loss < best_eval_loss:
                                            best_eval_loss = avg_valid_loss
                                            self.save("BEST_EVAL_LOSS")
                                        self.save("Epoch%02d" % (epoch + 1))

                                    dist.barrier()
                                else:
                                    # Skip validation
                                    print("Skip validation for Epoch%02d" % (epoch + 1))
                                    self.save("Epoch%02d" % (epoch + 1))

                                    dist.barrier()


                    # elif args.retrain_both:
                    elif args.retrain_type == "both":
                        pass

                elif args.use_fusion_net:
                    # print("+" * 50)
                    # print(args.use_fusion_net)
                    # print(self.args.use_fusion_net)
                    # print(self.args.fusion_net)
                    # print(self.args.retrain_large_model)
                    # print(self.args.retrain_small_model)
                    model_meta_fusion = model.get_model_meta(self.args.fusion_net)  # type: model.ModelMeta
                    model_conf_fusion, raw_model_conf_fusion = ap.parse_arch_config_from_args(model_meta_fusion, self.args)  # type: dict
                    fusion_net = model_meta_fusion.model_builder(model_conf=model_conf_fusion)  # type: torch.nn.module
                    self.fusion_net = fusion_net
                    # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict
                    LOSSES_NAME = self.args.LOSSES_NAME
                    optimizer = torch.optim.Adam(
                        self.fusion_net.parameters(),
                        # lr=float(args.learning_rate)
                        lr=float(args.lr)
                    )
                    return_large_pred_list = True
                    return_large_pred_point = not return_large_pred_list
                    if return_large_pred_list:
                        with torch.no_grad():
                            for step_i, batch in tqdm(enumerate(self.train_loader)):
                                results = self.large_language_model.generate_step(batch)
                                beam_outputs = model.generate(
                                    batch['input_ids'].to('cuda'),
                                    max_length=50,
                                    num_beams=20,
                                    no_repeat_ngram_size=0,
                                    num_return_sequences=20,
                                    early_stopping=True
                                )
                                generated_sents = self.large_language_model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                                for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                                    new_info = {}
                                    new_info['target_item'] = item[1]
                                    new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
                                    all_info.append(new_info)

                                # logits = self.small_recommendation_model({
                                #     key: value.to(device)
                                #     for key, value in batch.items()
                                #     # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                                #     if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE,
                                #                consts.FIELD_TRIGGER_SEQUENCE}
                                # }, generated_sents)
                                small_logits = self.small_recommendation_model({
                                    key: value.to(device)
                                    for key, value in batch.items()
                                    # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                                    if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE,
                                               consts.FIELD_TRIGGER_SEQUENCE}
                                })

                                criterion = nn.BCEWithLogitsLoss()
                                # loss = criterion(small_logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                                logits = fusion_net(generated_sents, small_logits)
                                loss = criterion(logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                    elif return_large_pred_point:
                        with torch.no_grad():
                            for i, batch in tqdm(enumerate(self.train_loader)):
                                results = model.generate_step(batch)
                                beam_outputs = model.generate(
                                    batch['input_ids'].to('cuda'),
                                    max_length=50,
                                    num_beams=20,
                                    no_repeat_ngram_size=0,
                                    num_return_sequences=20,
                                    early_stopping=True
                                )
                                generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
                                for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
                                    new_info = {}
                                    new_info['target_item'] = item[1]
                                    new_info['gen_item_list'] = generated_sents[j * 20: (j + 1) * 20]
                                    all_info.append(new_info)

                                # logits = self.small_recommendation_model({
                                #     key: value.to(device)
                                #     for key, value in batch.items()
                                #     # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                                #     if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE,
                                #                consts.FIELD_TRIGGER_SEQUENCE}
                                # }, generated_sents)
                                small_logits = self.small_recommendation_model({
                                    key: value.to(device)
                                    for key, value in batch.items()
                                    # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                                    if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE,
                                               consts.FIELD_TRIGGER_SEQUENCE}
                                })

                                criterion = nn.BCEWithLogitsLoss()
                                # loss = criterion(small_logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                                logits = fusion_net(generated_sents, small_logits)
                                loss = criterion(logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()

                if self.verbose:
                    pbar.close()

                dist.barrier()

                results = reduce_dict(epoch_results, average=False)
                if self.verbose:
                    train_loss = results['total_loss']
                    train_loss_count = results['total_loss_count']

                    avg_train_loss = train_loss / train_loss_count
                    losses_str = f"Train Loss: {avg_train_loss:.3f}\n"

                    for name, loss in results.items():
                        if name[-4:] == 'loss':
                            loss_count = int(results[name + '_count'])
                            if loss_count > 0:
                                avg_loss = loss / loss_count
                                losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                    losses_str += '\n'
                    print(losses_str)

                dist.barrier()

                if epoch > 10:
                    # Validation
                    valid_results = self.evaluate_epoch(epoch=epoch)

                    valid_results = reduce_dict(valid_results, average=False)
                    if self.verbose and step_i % 200:
                        valid_loss = valid_results['total_loss']
                        valid_loss_count = valid_results['total_loss_count']

                        avg_valid_loss = valid_loss / valid_loss_count
                        losses_str = f"Valid Loss: {avg_valid_loss:.3f}\n"

                        for name, loss in valid_results.items():
                            if name[-4:] == 'loss':
                                loss_count = int(valid_results[name + '_count'])
                                if loss_count > 0:
                                    avg_loss = loss / loss_count
                                    losses_str += f"{name} ({loss_count}): {avg_loss:.3f} "

                        losses_str += '\n'
                        print(losses_str)

                    dist.barrier()

                    if self.verbose:
                        # Save
                        if avg_valid_loss < best_eval_loss:
                            best_eval_loss = avg_valid_loss
                            self.save("BEST_EVAL_LOSS")
                        self.save("Epoch%02d" % (epoch + 1))

                    dist.barrier()
                else:
                    # Skip validation
                    print("Skip validation for Epoch%02d" % (epoch + 1))
                    self.save("Epoch%02d" % (epoch + 1))

                    dist.barrier()

        else:
            raise NotImplementedError

    def evaluate_epoch(self, epoch):
        if self.args.framework == "large_language_model":
            LOSSES_NAME = self.args.LOSSES_NAME

            epoch_results = {}
            for loss_name in LOSSES_NAME:
                epoch_results[loss_name] = 0.
                epoch_results[f'{loss_name}_count'] = 0

            self.large_language_model.eval()
            with torch.no_grad():
                if self.verbose:
                    loss_meter = LossMeter()
                    loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                    pbar = tqdm(total=len(self.val_loader), ncols=275)

                for step_i, batch in enumerate(self.val_loader):

                    if self.args.distributed:
                        results = self.large_language_model.module.valid_step(batch)
                    else:
                        results = self.large_language_model.valid_step(batch)

                    for k, v in results.items():
                        if k in epoch_results:
                            if isinstance(v, int):
                                epoch_results[k] += v
                            elif isinstance(v, torch.Tensor):
                                epoch_results[k] += v.item()

                    if self.verbose and step_i % 200:
                        desc_str = f'Valid Epoch {epoch} |'
                        for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):

                            if loss_name in results:
                                loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                            if len(loss_meter) > 0:
                                loss_count = epoch_results[f'{loss_name}_count']
                                desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'

                        pbar.set_description(desc_str)
                        pbar.update(1)
                    dist.barrier()

                if self.verbose:
                    pbar.close()
                dist.barrier()

                return epoch_results

        elif self.args.framework == "small_recommendation_model_base" or self.args.framework == "small_recommendation_model_duet":
            LOSSES_NAME = self.args.LOSSES_NAME

            # epoch_results = {}
            # for loss_name in LOSSES_NAME:
            #     epoch_results[loss_name] = 0.
            #     epoch_results[f'{loss_name}_count'] = 0

            # self.large_language_model.eval()
            # results = {}
            valid_results = {}
            self.small_recommendation_model.eval()
            valid_loss = 0
            valid_line_count = 0
            with torch.no_grad():
                if self.verbose:
                    loss_meter = LossMeter()
                    loss_meters = [LossMeter() for _ in range(len(LOSSES_NAME))]

                    pbar = tqdm(total=len(self.val_loader), ncols=275)

                # for step_i, batch in enumerate(self.val_loader):
                #
                #     if self.args.distributed:
                #         results = self.large_language_model.module.valid_step(batch)
                #     else:
                #         results = self.large_language_model.valid_step(batch)
                for step_i, batch in enumerate(self.valid_loader):
                    logits = self.small_recommendation_model({
                        key: value.to(device)
                        for key, value in batch.items()
                        # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
                        if key in {consts.FIELD_TARGET_ID, consts.FIELD_CLK_SEQUENCE}
                    })
                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(logits, batch[consts.FIELD_LABEL].view(-1, 1).to(device))
                    valid_loss += loss.detach() * len(batch[consts.FIELD_LABEL].view(-1, 1))
                    valid_line_count += len(batch[consts.FIELD_LABEL].view(-1, 1))
                    # for k, v in results.items():
                    #     if k in epoch_results:
                    #         if isinstance(v, int):
                    #             epoch_results[k] += v
                    #         elif isinstance(v, torch.Tensor):
                    #             epoch_results[k] += v.item()

                    if self.verbose and step_i % 200:
                        desc_str = f'Valid Epoch {epoch} |'
                        # for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                        #
                        #     if loss_name in results:
                        #         # loss_meter.update(results[f'{loss_name}'] / results[f'{loss_name}_count'])
                        #         loss_meter.update(loss)
                        #     if len(loss_meter) > 0:
                        #         loss_count = epoch_results[f'{loss_name}_count']
                        #         desc_str += f' {loss_name} ({loss_count}) {loss_meter.val:.3f}'
                        desc_str += f' {self.args.losses}_loss ({valid_line_count}) {loss_meter.val:.3f}'

                        pbar.set_description(desc_str)
                        pbar.update(1)
                    dist.barrier()

                valid_results["total_loss"] = valid_loss
                valid_results["total_loss_count"] = valid_line_count

                if self.verbose:
                    pbar.close()
                dist.barrier()

                # return epoch_results
                # return valid_loss
                return valid_results

        else:
            raise NotImplementedError


def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu}')
    # define the prompts used in training
    if args.train == 'yelp':
        # train_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        # 'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        # 'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
        # 'review': ['4-1', '4-2'],
        # 'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
        # }
        # if not args.framework == "collaboration":
        if args.framework == "large_language_model":
            train_task_list_all = {
                'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
                'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
                'review': ['4-1', '4-2'],
                'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
            }
            train_sample_numbers_all = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1,
                                        'traditional': (10, 5)}

            # if args.retrain_type == "small":
            #     train_sample_numbers_all = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1,
            #                                 'traditional': (1, 1)}

        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet" or args.framework == "collaboration":
            train_task_list_all = {
                'sequential': ['2-3', '2-13'],
                'traditional': ['5-5', '5-8']
            }
            train_sample_numbers_all = {
                'sequential': (1, 1, 1),
                'traditional': (1, 1)
            }
        else:
            raise NotImplementedError

        if args.framework == "large_language_model":
            if args.type != "all":
                train_task_list = {args.type: train_task_list_all[args.type]}
                train_sample_numbers = {args.type: train_sample_numbers_all[args.type]}
            else:
                train_task_list = train_task_list_all
                train_sample_numbers = train_sample_numbers_all

        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet":
            # train_task_list = {'Sequential_ctr': []}
            # train_sample_numbers = {'Sequential_ctr': (5, 5, 10)}
            # train_sample_numbers_all = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1,
            #                             'traditional': (10, 5)}
            train_task_list = {args.type: []}
            train_sample_numbers = {args.type: train_sample_numbers_all[args.type]}

        elif args.framework == "collaboration":
            if args.type != "all":
                train_task_list = {args.type: train_task_list_all[args.type]}
                train_sample_numbers = {args.type: train_sample_numbers_all[args.type]}
            else:
                train_task_list = train_task_list_all_large
                train_sample_numbers = train_sample_numbers_all

        else:
            raise NotImplementedError
    else:
        # train_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        # 'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        # 'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
        # 'review': ['4-1', '4-2', '4-3'],
        # 'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
        # }
        if not args.framework == "collaboration":
            train_task_list_all = {
                'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
                'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
                'review': ['4-1', '4-2', '4-3'],
                'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
            }
            train_sample_numbers_all = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1,
                                        'traditional': (10, 5)}

            # if args.retrain_type == "small":
            #     train_sample_numbers_all = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1,
            #                                 'traditional': (1, 1)}

        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet" or args.framework == "collaboration":
            train_task_list_all = {
                'sequential': ['2-3', '2-13'],
                'traditional': ['5-5', '5-8']
            }
            train_sample_numbers_all = {
                'sequential': (1, 1, 1),
                'traditional': (1, 1)
            }
        else:
            raise NotImplementedError

        if args.framework == "large_language_model":
            if args.type != "all":
                train_task_list = {args.type: train_task_list_all[args.type]}
                train_sample_numbers = {args.type: train_sample_numbers_all[args.type]}
            else:
                train_task_list = train_task_list_all
                train_sample_numbers = train_sample_numbers_all

        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet":
            # train_task_list = {'Sequential_ctr': []}
            # train_sample_numbers = {'Sequential_ctr': (5, 5, 10)}
            # train_task_list = {args.losses: []}
            # train_sample_numbers = {args.losses: train_sample_numbers_all[args.type]}
            train_task_list = {args.type: []}
            train_sample_numbers = {args.type: train_sample_numbers_all[args.type]}


        # elif args.framework == "small_recommendation_model_duet":
        #     train_task_list = {'Sequential_ctr': []}
        #     train_sample_numbers = {'Sequential_ctr': (5, 5, 10)}

        elif args.framework == "collaboration":
            if args.type != "all":
                train_task_list = {args.type: train_task_list_all[args.type]}
                train_sample_numbers = {args.type: train_sample_numbers_all[args.type]}
            else:
                train_task_list = train_task_list_all
                train_sample_numbers = train_sample_numbers_all

        else:
            raise NotImplementedError
    # define sampling numbers for each group of personalized prompts (see pretrain_data.py)
    # if greater than 1, a data sample will be used for multiple times with different prompts in certain task family
    # train_sample_numbers = {'rating': 1, 'sequential': (5, 5, 10), 'explanation': 1, 'review': 1, 'traditional': (10, 5)}
    # train_sample_numbers = {'Sequential_ctr': (5, 5, 10)}
    train_loader = get_loader(
        args,
        train_task_list,
        train_sample_numbers,
        split=args.train,
        mode='train',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )

    print(f'Building val loader at GPU {gpu}')
    # define the prompts used in validation
    if args.valid == 'yelp':
        # val_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        # 'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        # 'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
        # 'review': ['4-1', '4-2'],
        # 'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
        # }
        if args.framework == "large_language_model":
            val_task_list_all = {
                'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
                'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9'],
                'review': ['4-1', '4-2'],
                'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
            }
            val_sample_numbers_all = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1,
                                      'traditional': (1, 1)}
        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet" or args.framework == "collaboration":
            val_task_list_all = {
                'sequential': ['2-3', '2-13'],
                'traditional': ['5-5', '5-8']
            }
            val_sample_numbers_all = {
                'sequential': (1, 1, 1),
                'traditional': (1, 1)
            }
        else:
            raise NotImplementedError

        if args.framework == "large_language_model":
            if args.type != "all":
                val_task_list = {args.type: val_task_list_all[args.type]}
                val_sample_numbers = {args.type: val_sample_numbers_all[args.type]}
            else:
                val_task_list = val_task_list_all
                val_sample_numbers = val_sample_numbers_all

        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet":
            # val_task_list = {'Sequential_ctr': []}
            # val_sample_numbers = {'Sequential_ctr': (1, 1, 1)}
            # val_task_list = {args.losses: []}
            # val_sample_numbers = {args.losses: (1, 1, 1)}
            val_task_list = {args.type: []}
            val_sample_numbers = {args.type: val_sample_numbers_all[args.type]}

        elif args.framework == "collaboration":
            if args.type != "all":
                val_task_list = {args.type: val_task_list_all[args.type]}
                val_sample_numbers = {args.type: val_sample_numbers_all[args.type]}
            else:
                val_task_list = val_task_list_all
                val_sample_numbers = val_sample_numbers_all

        else:
            raise NotImplementedError
    else:
        # val_task_list = {'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
        # 'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
        # 'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
        # 'review': ['4-1', '4-2', '4-3'],
        # 'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
        # }
        if args.framework == "large_language_model":
            val_task_list_all = {
                'rating': ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9'],
                'sequential': ['2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9', '2-10', '2-11', '2-12'],
                'explanation': ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11'],
                'review': ['4-1', '4-2', '4-3'],
                'traditional': ['5-1', '5-2', '5-3', '5-4', '5-5', '5-6', '5-7']
            }
            val_sample_numbers_all = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1,
                                      'traditional': (1, 1)}
        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet" or args.framework == "collaboration":
            val_task_list_all = {
                'sequential': ['2-3', '2-13'],
                'traditional': ['5-5', '5-8']
            }
            val_sample_numbers_all = {
                'sequential': (1, 1, 1),
                'traditional': (1, 1)
            }
        else:
            raise NotImplementedError

        if args.framework == "large_language_model":
            if args.type != "all":
                val_task_list = {args.type: val_task_list_all[args.type]}
                val_sample_numbers = {args.type: val_sample_numbers_all[args.type]}
            else:
                val_task_list = val_task_list_all
                val_sample_numbers = val_sample_numbers_all

        elif args.framework == "small_recommendation_model_base" or args.framework == "small_recommendation_model_duet":
            # val_task_list = {'Sequential_ctr': []}
            # val_sample_numbers = {'Sequential_ctr': (1, 1, 1)}
            # val_task_list = {args.losses: []}
            # val_sample_numbers = {args.losses: (1, 1, 1)}
            val_task_list = {args.type: []}
            val_sample_numbers = {args.type: val_sample_numbers_all[args.type]}

        elif args.framework == "collaboration":
            if args.type != "all":
                val_task_list = {args.type: val_task_list_all[args.type]}
                val_sample_numbers = {args.type: val_sample_numbers_all[args.type]}
            else:
                val_task_list = val_task_list_all
                val_sample_numbers = val_sample_numbers_all

        else:
            raise NotImplementedError
    # val_sample_numbers = {'rating': 1, 'sequential': (1, 1, 1), 'explanation': 1, 'review': 1, 'traditional': (1, 1)}
    # val_sample_numbers = {'Sequential_ctr': (1, 1, 1)}
    val_loader = get_loader(
        args,
        val_task_list,
        val_sample_numbers,
        split=args.valid,
        mode='val',
        batch_size=args.batch_size,
        workers=args.num_workers,
        distributed=args.distributed
    )

    trainer = Trainer(args, train_loader, val_loader, train=True)
    trainer.train()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    if args.local_rank in [0, -1]:
        print(args)

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
    if args.local_rank in [0, -1]:
        print(LOSSES_NAME)
    LOSSES_NAME.append('total_loss')

    args.LOSSES_NAME = LOSSES_NAME

    comments = []
    dsets = []
    if 'toys' in args.train:
        dsets.append('toys')
    if 'beauty' in args.train:
        dsets.append('beauty')
    if 'sports' in args.train:
        dsets.append('sports')
    if 'yelp' in args.train:
        dsets.append('yelp')
    comments.append(''.join(dsets))
    if args.backbone:
        comments.append(args.backbone)
    comments.append(''.join(args.losses.split(',')))
    if args.comment != '':
        comments.append(args.comment)
    comment = '_'.join(comments)

    from datetime import datetime

    current_time = datetime.now().strftime('%b%d_%H-%M')

    project_dir = Path(__file__).resolve().parent.parent

    if args.local_rank in [0, -1]:
        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'
        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
