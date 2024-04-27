import torch.nn
import torch.nn as nn

from module import encoder, common, initializer
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("sasrec_fusion", MetaType.ModelBuilder)
class SasRec(nn.Module):
    def __init__(self, model_conf):
        super(SasRec, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)

        self._position_embedding = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension
        )

        self._id_encoder = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension
        )

        self._target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension] * 2, [torch.nn.Tanh, None]
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )

        self._transformer = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4*model_conf.id_dimension,
            dropout=0
        )

        initializer.default_weight_init(self._transformer.self_attn.in_proj_weight)
        initializer.default_weight_init(self._transformer.self_attn.out_proj.weight)
        initializer.default_bias_init(self._transformer.self_attn.in_proj_bias)
        initializer.default_bias_init(self._transformer.self_attn.out_proj.bias)

        initializer.default_weight_init(self._transformer.linear1.weight)
        initializer.default_bias_init(self._transformer.linear1.bias)
        initializer.default_weight_init(self._transformer.linear2.weight)
        initializer.default_bias_init(self._transformer.linear2.bias)

        self._classifier = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._classifier_large = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._fusion_layer = nn.Linear(2, 1, bias=True)

    def forward(self, features, large_pred=None, loss="traditional", device=None):
        # print("=" * 50)
        # for key, value in features.items():
        #     print("-" * 50)
        #     print(key)
        #     # print(value)
        #     print(value.size())
        # print("-" * 50)
        # print("large_pred")
        # print(large_pred.shape)
        # print(torch.from_numpy(large_pred).size())

        # Encode target item
        # B * D
        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        # print("size_size_size")
        # print(features)
        # print(target_embed)
        # print(target_embed.size())
        target_embed = self._target_trans(target_embed)
        # print(target_embed.size())
        # Encode user historical behaviors
        # if len(large_pred) > 1:
        if type(large_pred) == "list":
            large_pred.reverse()
            large_pred = torch.from_numpy(np.array([int(i) for i in large_pred]))

        with torch.no_grad():
            click_seq = features[consts.FIELD_CLK_SEQUENCE]
            # print(click_seq)
            batch_size = int(click_seq.shape[0])
            large_pred_batch_size = int(large_pred.size()[0])
            # print(batch_size)
            # print(click_seq.size())
            # print(target_embed.size())
            # B * L
            positions = torch.arange(0, int(click_seq.shape[1]), dtype=torch.int32).to(click_seq.device)
            positions = torch.tile(positions.unsqueeze(0), [batch_size, 1])
            mask = torch.not_equal(click_seq, 0)
            # B
            seq_length = torch.maximum(torch.sum(mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
            seq_length = seq_length.to(torch.long)

            large_pred_embed_position = torch.arange(0, int(large_pred.shape[1]), dtype=torch.int32).to(
                large_pred.device)
            # large_pred_embed_position = torch.tile(large_pred_embed_position.unsqueeze(0), [batch_size, 1])
            large_pred_embed_position = torch.tile(large_pred_embed_position.unsqueeze(0), [large_pred_batch_size, 1])
            large_pred_mask = torch.not_equal(large_pred, 0)

            large_pred_length = torch.maximum(torch.sum(large_pred_mask.to(torch.int32), dim=1) - 1,
                                              torch.Tensor([0]).to(device=large_pred_mask.device))
            large_pred_length = large_pred_length.to(torch.long)
            # if loss == "traditional":

            # # if len(large_pred) > 1:
            # if type(large_pred) == "list":
            #     large_pred_embed_position = torch.arange(0, int(large_pred.shape[1]), dtype=torch.int32).to(large_pred.device)
            #     large_pred_embed_position = torch.tile(large_pred_embed_position.unsqueeze(0), [batch_size, 1])
            #     large_pred_mask = torch.not_equal(large_pred, 0)
            #
            #     large_pred_length = torch.maximum(torch.sum(large_pred_mask.to(torch.int32), dim=1) - 1, torch.Tensor([0]).to(device=large_pred_mask.device))
            #     large_pred_length = large_pred_length.to(torch.long)

                # B * L * D
        hist_embed = self._id_encoder(click_seq)
        hist_pos_embed = self._position_embedding(positions)
        hist_embed = self._seq_trans(hist_embed + hist_pos_embed)

        atten_embed = self._transformer(
            torch.swapaxes(hist_embed, 0, 1)
        )
        user_state = torch.swapaxes(atten_embed, 0, 1)[range(batch_size), seq_length, :]

        user_embedding = self._classifier(user_state)

        # if loss == "traditional":
        # if len(large_pred) > 1:

        # if type(large_pred) == "list":
        #     large_pred_embed = self._id_encoder(large_pred)
        #     large_pred_pos_embed = self._position_embedding(large_pred_embed_position)
        #     large_pred_embed = self._seq_trans(large_pred_embed + large_pred_pos_embed)
        #
        #     large_pred_atten_embed = self._transformer(
        #         torch.swapaxes(large_pred_embed, 0, 1)
        #     )
        #     large_pred_user_state = torch.swapaxes(large_pred_atten_embed, 0, 1)[range(batch_size), large_pred_length, :]
        #
        #     large_pred_user_embedding = self._classifier(large_pred_user_state)

        large_pred_embed = self._id_encoder(large_pred)
        large_pred_pos_embed = self._position_embedding(large_pred_embed_position)
        # large_pred_user_embedding = torch.mean(large_pred_embed, dim=1)
        large_pred_embed = self._seq_trans(large_pred_embed + large_pred_pos_embed)
        large_pred_atten_embed = self._transformer(
            torch.swapaxes(large_pred_embed, 0, 1)
        )
        large_pred_user_state = torch.swapaxes(large_pred_atten_embed, 0, 1)[range(large_pred_batch_size), large_pred_length, :]

        large_pred_user_embedding = self._classifier_large(large_pred_user_state)

        large_pred_user_embedding = large_pred_user_embedding.repeat_interleave(5, dim=0)
        # large_pred_user_embedding = torch.mean(large_pred_embed)
        # print("///")
        # print("---")
        # print(large_pred_embed)
        # print(large_pred_embed.size())
        # print(torch.mean(large_pred_embed).size())
        # print(torch.mean(large_pred_embed, dim=0).size())
        # print(torch.mean(large_pred_embed, dim=1).size())
        # print(torch.mean(large_pred_embed, dim=2).size())
        # print(torch.mean(large_pred_embed, dim=-1).size())
        # print("---")
        # print(user_embedding.size())
        # print(large_pred_user_embedding.size())
        # print(target_embed.size())
        # print(torch.cat(
        #         [torch.sum(user_embedding * target_embed, dim=1, keepdim=True),
        #         torch.sum(large_pred_user_embedding * target_embed, dim=1, keepdim=True)], dim=0
        #     ).size())
        # large_pred_user_embedding = large_pred_user_embedding.repeat_interleave(5, dim=0)
        # print(torch.cat(
        #         [torch.sum(user_embedding * target_embed, dim=1, keepdim=True),
        #         torch.sum(large_pred_user_embedding * target_embed, dim=1, keepdim=True)], dim=1
        #     ).size())
        # print(torch.cat(
        #         [torch.sum(user_embedding * target_embed, dim=1, keepdim=True),
        #         torch.sum(large_pred_user_embedding * target_embed, dim=1, keepdim=True)], dim=2
        #     ).size())
        # print(large_pred_user_embedding.size())
        return self._fusion_layer(
            torch.cat(
                [torch.sum(user_embedding * target_embed, dim=1, keepdim=True),
                torch.sum(large_pred_user_embedding * target_embed, dim=1, keepdim=True),], dim=1
            )
        )
