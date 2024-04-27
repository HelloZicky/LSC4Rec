"""
Common modules
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import initializer


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i - 1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())

        self.net = torch.nn.Sequential(*modules)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Linear(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, bias):
        super(Linear, self).__init__()
        self.net = torch.nn.Linear(in_dimension, out_dimension, bias)
        initializer.default_weight_init(self.net.weight)
        if bias:
            initializer.default_weight_init(self.net.bias)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class HyperNetwork_FC(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        return x

from . import config
import os, sys
sys.path.append("..")
from src.pretrain_model import P5Pretraining

class HyperNetwork_FC_Large(nn.Module):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_Large, self).__init__()
        self.ckpt_path = "/data/zhantianyu/LLM/P5_DCC_Rec_Big/checkpoint_large/beauty/t5-small_sequential/Epoch10.pth"
        self.args = self.create_args()
        self.config = config.create_config(self.args)
        self.tokenizer = config.create_tokenizer(self.args)
        model_class = P5Pretraining
        self.large_language_model = config.create_model(model_class,self.args.backbone,self.config)
        self.large_language_model = self.large_language_model.cuda()
        if 'p5' in self.args.tokenizer:
            self.large_language_model.resize_token_embeddings(self.tokenizer.vocab_size)
        self.large_language_model.tokenizer = self.tokenizer
        state_dict = config.load_state_dict(self.ckpt_path, 'cpu')
        self.large_language_model.load_state_dict(state_dict, strict=False)
        self.large_language_model = self.large_language_model.get_encoder()
        self.large_language_model = self.large_language_model.cuda()
        for item in self.large_language_model.parameters():
            item.requires_grad = False

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._mlp_trans_first = StackedDense(
            512,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def create_args(self):
        args = config.DotDict()
        args.backbone = '/data/zhantianyu/LLM/llm/t5-small'
        args.losses = "sequential"
        args.local_rank = 0
        args.comment = ''
        args.train_topk = -1
        args.valid_topk = -1
        args.dropout = 0.1

        args.tokenizer = 'p5'
        args.max_text_length = 30
        args.do_lower_case = False
        args.word_mask_rate = 0.15
        args.gen_max_length = 64

        args.whole_word_embed = True
        return args

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # template = "According to the purchase history of user : \n {} \n Can you recommend the next possible item to the user ?"
        z = self.tokenizer.batch_encode_plus(z,padding='max_length',return_tensors = 'pt', truncation=True, max_length=self.args.max_text_length)
        mask = z['attention_mask'].cuda()
        seq_length = torch.maximum(torch.sum(mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
        seq_length = seq_length.to(torch.long)
        z = z['input_ids'].cuda()#z:(32,10,512)
        z = self.large_language_model(z)[0]#z:(32,10,512)
        z = self._mlp_trans_first(z)#z:(32,10,32)
        
        user_state, _ = self._gru_cell(z)#z:(32,10,32)
        user_state = user_state[range(user_state.shape[0]), seq_length, :]#(32,10,32)->(32,32)
        z = self._mlp_trans(user_state)#z:(32,32)
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index] #weight:(32,1024)

            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]#bias:(32,32)

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias #x:(32,32)
            x = self.modules[index](x) if index < len(self.modules) else x

        return x
