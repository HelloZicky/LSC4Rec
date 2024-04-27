import torch.nn
import torch.nn as nn

from module import encoder, common, initializer
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("gru4rec_fusion", MetaType.ModelBuilder)
class GRU4Rec(nn.Module):
    def __init__(self, model_conf):
        super(GRU4Rec, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)

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

        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self._classifier = common.StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._fusion_layer = nn.Linear(2, 1, bias=True)

    def forward(self, features, large_pred=None):
        # Encode target item
        # B * D
        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            mask = torch.not_equal(features[consts.FIELD_CLK_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            seq_length = torch.maximum(torch.sum(mask, dim=1) - 1, torch.Tensor([0]).to(device=mask.device))
            seq_length = seq_length.to(torch.long)

        # B * L * D
        hist_embed = self._id_encoder(features[consts.FIELD_CLK_SEQUENCE])
        hist_embed = self._seq_trans(hist_embed)

        # Get embedding of last step
        user_state, _ = self._gru_cell(hist_embed)
        user_state = user_state[range(user_state.shape[0]), seq_length, :]

        user_embedding = self._classifier(user_state)

        large_pred_embed = self._id_encoder(large_pred)
        large_pred_user_embedding = torch.mean(large_pred_embed, dim=1)
        # print("-" * 50)
        # print("large_pred_embed.size() ", large_pred_embed.size())
        # print("large_pred_user_embedding.size() ", large_pred_user_embedding.size())
        # print("large_pred_embed ", large_pred_embed)
        # print("large_pred_user_embedding ", large_pred_user_embedding)
        # print("large_pred_user_embedding.repeat_interleave(5, dim=0).size() ",
        #       large_pred_user_embedding.repeat_interleave(5, dim=0).size())

        large_pred_user_embedding = large_pred_user_embedding.repeat_interleave(5, dim=0)

        # return torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
        return self._fusion_layer(
            torch.cat(
                [torch.sum(user_embedding * target_embed, dim=1, keepdim=True),
                torch.sum(large_pred_user_embedding * target_embed, dim=1, keepdim=True),], dim=1
            )
        )