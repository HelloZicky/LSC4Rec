import torch
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import pprint

# class DotDict(dict):
#     def __init__(self, **kwds):
#         self.update(kwds)
#         self.__dict__ = self
#
#
# args = DotDict()

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=2022, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--submit', action='store_true')

    # Checkpoint
    parser.add_argument('--output', type=str, default='test/pretrain')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--local_rank', type=int, default=0)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default='p5')
    parser.add_argument('--whole_word_embed', action='store_true')

    parser.add_argument('--max_text_length', type=int, default=512)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.1)

    # parser.add_argument("--losses", default='rating,sequential,review,metadata,recommend', type=str)
    parser.add_argument("--losses", default='rating,sequential,explanation,review,traditional', type=str)
    parser.add_argument('--log_train_accuracy', action='store_true')

    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=64)

    # Data
    parser.add_argument('--do_lower_case', action='store_true')

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    return args


# args = parse_args()
def reset_params(args):
    args.distributed = False
    args.multiGPU = True
    args.fp16 = True
    # args.train = "beauty"
    # args.valid = "beauty"
    # args.test = "beauty"
    args.batch_size = 16
    # args.optim = 'adamw'
    # args.warmup_ratio = 0.05
    # args.lr = 1e-3
    # args.num_workers = 4
    # args.clip_grad_norm = 1.0
    # args.losses = 'rating,sequential,explanation,review,traditional'
    # args.backbone = 't5-small'  # small or base
    # args.backbone = '/data/lvzheqi/lvzheqi/ICLR2023/LLM/llm/t5-small'  # small or base
    args.backbone = '/data/home/lzq/lvzheqi/ICLR2023/LLM/llm/t5-small'  # small or base
    # args.output = 'snap/beauty-small'
    # args.epoch = 10
    # args.local_rank = 0

    # args.comment = ''
    # args.train_topk = -1
    # args.valid_topk = -1
    # args.dropout = 0.1

    # args.tokenizer = 'p5'
    # args.max_text_length = 512
    args.do_lower_case = False
    # args.word_mask_rate = 0.15
    # args.gen_max_length = 64

    # args.weight_decay = 0.01
    # args.adam_eps = 1e-6
    # args.gradient_accumulation_steps = 1

    '''
    Set seeds
    '''
    args.seed = 2022

    '''
    Whole word embedding
    '''
    args.whole_word_embed = True

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    gpu = 0  # Change GPU ID
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    torch.cuda.set_device('cuda:{}'.format(gpu))
    # args.load = "snap/beauty-small.pth"
