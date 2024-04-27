import torch
from packaging import version
from func_utils import *
from func_set_params import parse_args, reset_params
import random
import numpy as np
import torch.backends.cudnn as cudnn

args = parse_args()
print(args)
reset_params(args)
print(args)
# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

cudnn.benchmark = True

if args.local_rank in [0, -1]:
    print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

comments = []
dsets = []
if 'toys' in args.train:
    dsets.append('toys')
if 'beauty' in args.train:
    dsets.append('beauty')
if 'sports' in args.train:
    dsets.append('sports')
comments.append(''.join(dsets))
if args.backbone:
    comments.append(args.backbone)
comments.append(''.join(args.losses.split(',')))
if args.comment != '':
    comments.append(args.comment)
comment = '_'.join(comments)
print(comment)
print(dsets)

from func_build_model import
model = func_build_model
from func_evaluate_sequential import evaluate_sequential
evaluate_sequential(args, model)
