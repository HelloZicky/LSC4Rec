import torch.nn
import torch.nn as nn

from module import attention, encoder, common
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("meta_din", MetaType.ModelBuilder)
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

        # self._classifier = common.StackedDense(
        #     model_conf.id_dimension * 2,
        #     model_conf.classifier + [1],
        #     ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        # )

        self._meta_classifier_param_list = common.HyperNetwork_FC(
            # model_conf.id_dimension * 2,
            model_conf.id_dimension,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None],
            batch=True,
            trigger_sequence_len=30,
            model_conf=model_conf,
            expand=True
        )
        
    def __setitem__(self, k, v):
        self.k = v

    def forward(self, features, fig1=False):
        # Encode target item
        # B * D
        # print("-" * 50)
        # print(features)
        trigger_embed = self._id_encoder(features[consts.FIELD_TRIGGER_SEQUENCE])
        trigger_embed = self._seq_trans(trigger_embed)

        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            click_mask = torch.not_equal(features[consts.FIELD_CLK_SEQUENCE], 0).to(dtype=torch.int32)
            trigger_mask = torch.not_equal(features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            seq_length = torch.maximum(torch.sum(click_mask, dim=1) - 1, torch.Tensor([0]).to(device=click_mask.device))
            seq_length = seq_length.to(torch.long)
            trigger_seq_length = torch.maximum(torch.sum(trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=trigger_mask.device))
            trigger_seq_length = trigger_seq_length.to(torch.long)

        # B * L * D
        hist_embed = self._id_encoder(features[consts.FIELD_CLK_SEQUENCE])
        hist_embed = self._seq_trans(hist_embed)

        # Target attention
        atten_aggregated_embed = self._target_attention(
            target_key=target_embed,
            item_keys=hist_embed,
            item_values=hist_embed,
            mask=click_mask
        )

        classifier_input = torch.cat([target_embed, atten_aggregated_embed], dim=1)

        if fig1:
            return self._meta_classifier_param_list(classifier_input, hist_embed,
                                                    classifier_input.size()[0],
                                                    trigger_seq_length=seq_length
                                                    ), \
                   self._meta_classifier_param_list(classifier_input, trigger_embed,
                                                classifier_input.size()[0],
                                                trigger_seq_length=trigger_seq_length
                                                )
        return self._meta_classifier_param_list(classifier_input, hist_embed,
                                                classifier_input.size()[0],
                                                trigger_seq_length=seq_length
                                                )

@model("meta_din_large", MetaType.ModelBuilder)
class DeepInterestNetwork_large(nn.Module):
    def __init__(self, model_conf):
        super(DeepInterestNetwork_large, self).__init__()

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

        # self._classifier = common.StackedDense(
        #     model_conf.id_dimension * 2,
        #     model_conf.classifier + [1],
        #     ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        # )

        self._meta_classifier_param_list = common.HyperNetwork_FC_Large(
            # model_conf.id_dimension * 2,
            model_conf.id_dimension,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None],
            batch=True,
            trigger_sequence_len=30,
            model_conf=model_conf,
            expand=True
        )
        
    def __setitem__(self, k, v):
        self.k = v

    def forward(self, features, fig1=False):
        # Encode target item
        # B * D
        # print("-" * 50)
        # print(features)
        trigger_embed = self._id_encoder(features[consts.FIELD_TRIGGER_SEQUENCE])
        trigger_embed = self._seq_trans(trigger_embed)

        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            click_mask = torch.not_equal(features[consts.FIELD_CLK_SEQUENCE], 0).to(dtype=torch.int32)
            trigger_mask = torch.not_equal(features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            seq_length = torch.maximum(torch.sum(click_mask, dim=1) - 1, torch.Tensor([0]).to(device=click_mask.device))
            seq_length = seq_length.to(torch.long)
            trigger_seq_length = torch.maximum(torch.sum(trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=trigger_mask.device))
            trigger_seq_length = trigger_seq_length.to(torch.long)

        # B * L * D
        hist_embed = self._id_encoder(features[consts.FIELD_CLK_SEQUENCE])
        hist_embed = self._seq_trans(hist_embed)

        # Target attention
        atten_aggregated_embed = self._target_attention(
            target_key=target_embed,
            item_keys=hist_embed,
            item_values=hist_embed,
            mask=click_mask
        )

        classifier_input = torch.cat([target_embed, atten_aggregated_embed], dim=1)

        if fig1:
            return self._meta_classifier_param_list(classifier_input, features['click_string'],
                                                    classifier_input.size()[0],
                                                    trigger_seq_length=seq_length
                                                    ), \
                   self._meta_classifier_param_list(classifier_input, features[consts.FIELD_TRIGGER_SEQUENCE],
                                                classifier_input.size()[0],
                                                trigger_seq_length=trigger_seq_length
                                                )
        return self._meta_classifier_param_list(classifier_input, features['click_string'],
                                                classifier_input.size()[0],
                                                trigger_seq_length=seq_length
                                                )


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.params = nn.ModuleDict({"conv": self.conv})
        self.params2 = nn.ModuleDict({"conv": self.conv})

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    m = TestModule()
    for idx,p in enumerate(m.parameters()):
        print(idx, ":", p)
