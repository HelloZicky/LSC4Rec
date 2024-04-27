import torch.nn
import torch.nn as nn

from module import attention, encoder, common
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("din", MetaType.ModelBuilder)
class DeepInterestNetwork(nn.Module):
    def __init__(self, model_conf):
        super(DeepInterestNetwork, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)
        self._id_encoder = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension,
        )
        self._target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )

        self._target_attention = attention.TargetAttention(
            key_dimension=model_conf.id_dimension,
            value_dimension=model_conf.id_dimension,
        )

        self._classifier = common.StackedDense(
            model_conf.id_dimension * 2,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        )
        
    def __setitem__(self, k, v):
        self.k = v

    def forward(self, features):
        # Encode target item
        # B * D
        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            mask = torch.not_equal(features[consts.FIELD_CLK_SEQUENCE], 0).to(dtype=torch.float32)
        # B * L * D
        hist_embed = self._id_encoder(features[consts.FIELD_CLK_SEQUENCE])
        hist_embed = self._seq_trans(hist_embed)

        # Target attention
        atten_aggregated_embed = self._target_attention(
            target_key=target_embed,
            item_keys=hist_embed,
            item_values=hist_embed,
            mask=mask
        )

        classifier_input = torch.cat([target_embed, atten_aggregated_embed], dim=1)
        return self._classifier(classifier_input)


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.params = nn.ModuleDict({"conv": self.conv})
        self.params2 = nn.ModuleDict({"conv": self.conv})

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    # m = TestModule()
    # for idx,p in enumerate(m.parameters()):
    #     print(idx, ":", p)
    from thop import profile

    # input = torch.randn(1, 3, 490, 490)
    # input = torch.randn(1, 3, 224, 224)
    input = {
        consts.FIELD_USER_ID: [item[0] for item in data],
        consts.FIELD_TARGET_ID: torch.from_numpy(np.stack([item[1] for item in data], axis=0)),
        consts.FIELD_CLK_SEQUENCE: torch.from_numpy(np.stack([item[2] for item in data], axis=0)),
        consts.FIELD_TRIGGER_SEQUENCE: torch.from_numpy(np.stack([item[3] for item in data], axis=0)),
        consts.FIELD_LABEL: torch.from_numpy(np.stack([item[4] for item in data], axis=0))
    }
    output_size = 8
    net = MobileNetV3_Large(output_size=output_size)
    flops, params = profile(net, inputs=(input,))
    print("flops={}B, params={}M".format(flops / 1e9, params / 1e6))
    net = MobileNetV3_Small(output_size=output_size)
    flops, params = profile(net, inputs=(input,))
    print("flops={}B, params={}M".format(flops / 1e9, params / 1e6))