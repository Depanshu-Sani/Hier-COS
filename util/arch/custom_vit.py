import torch
import torch.nn as nn
import torch.nn.functional as F
from util.arch.HAF_resnet import NarrowResidualTransformationHead
from transformers import (
    ViTForImageClassification,
    ViTConfig,
    ViTModel,
)
from transformers.modeling_outputs import (
    ImageClassifierOutput,
    BaseModelOutputWithPooling,
)
from typing import Optional, Union, Tuple


class ViT(nn.Module):
    def __init__(self, model, feature_size=768, num_classes=100, standard_vit=True):
        super(ViT, self).__init__()

        for param in model.parameters():
            param.requires_grad = False
        self.features_2 = model
        self.classifier_3 = nn.Linear(feature_size, num_classes)
        self.standard_vit = standard_vit

    def penultimate_feature(self, x, targets="ignored"):
        x = self.features_2(x)
        return x

    def forward(self, x, targets="ignored"):
        x = self.features_2(x)
        if self.standard_vit:
            x = x.pooler_output
        x = self.classifier_3(x)
        return x


class HAFrameViTNet(nn.Module):
    def __init__(self, model, num_classes, feature_size=768, standard_vit=True):
        super(HAFrameViTNet, self).__init__()

        for param in model.parameters():
            param.requires_grad = False
        self.features_2 = model
        self.bn2 = nn.BatchNorm1d(feature_size, momentum=0.001)
        self.projection = NarrowResidualTransformationHead(feature_size,
                                                           num_classes,
                                                           'prelu')
        self.standard_vit = standard_vit

    def forward(self, x, target="ignored"):
        out = self.features_2(x)
        if self.standard_vit:
            out = out.pooler_output
        # add bn and transformation module
        out = self.bn2(out)
        out = self.projection(out)
        return out


class HAFrameViT(nn.Module):
    def __init__(self, model, num_classes, haf_cls_weights=None):
        super(HAFrameViT, self).__init__()

        self.features_2 = model
        self.classifier_3 = nn.Linear(num_classes, num_classes, bias=True)
        if haf_cls_weights is not None:
            with torch.no_grad():
                self.classifier_3.weight = nn.Parameter(torch.Tensor(haf_cls_weights))
                self.classifier_3.weight.requires_grad_(False)
                self.classifier_3.bias = nn.Parameter(torch.zeros([num_classes, ]))
                self.classifier_3.bias.requires_grad_(False)

    def forward(self, x, target="ignored"):
        x = self.features_2(x)
        x = self.classifier_3(x)
        return x

    def penultimate_feature(self, x):
        x = self.features_2(x)
        return x


class CustomViTModel(ViTModel):
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = sequence_output[:, 1:, :].mean(dim=1)

        sequence_output = self.layernorm(sequence_output)
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            head_outputs = (
                (sequence_output, pooled_output)
                if pooled_output is not None
                else (sequence_output,)
            )
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class CustomViTForImageClassification(ViTForImageClassification):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = CustomViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        return outputs[0]