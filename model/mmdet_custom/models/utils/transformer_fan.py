import math
import torch
import torch.nn as nn
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, FEEDFORWARD_NETWORK)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer)
from mmcv.runner.base_module import BaseModule
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
import copy
import warnings


class DilationPredictor(nn.Module):
    def __init__(self, dim, num_scales, temperature=1):
        super(DilationPredictor, self).__init__()

        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Conv2d(dim, num_scales, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(num_scales),
            nn.Conv2d(num_scales, num_scales, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1)
        x = x / self.temperature
        x = x.softmax(dim=-1)
        return x


class DilatedAvgPooling(nn.Module):
    def __init__(self, dilation=0):
        super(DilatedAvgPooling, self).__init__()
        self.dilation = dilation
        if dilation>0:
            self.unfold = nn.Unfold(3, dilation=dilation, padding=dilation, stride=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.dilation > 0:
            x = self.unfold(x)
            new_shape = (B, C, 9, H, W)
            x = x.reshape(new_shape).mean(dim=2)
        return x.unsqueeze(-1)


class TAP(nn.Module):
    def __init__(self, embed_dims, num_scales=4):
        super().__init__()
        self.token_pools = nn.ModuleList([])
        for i in range(num_scales):
            self.token_pools.append(DilatedAvgPooling(i))
        self.dilation_predictor = DilationPredictor(embed_dims, len(self.token_pools))
        self.num_scales = num_scales

    def forward(self, x):
        mask = self.dilation_predictor(x) # B 1 H W K
        mixed_scales = []
        for i in range(self.num_scales):
            mixed_scales.append(self.token_pools[i](x))
        x_attn = torch.cat(mixed_scales, dim=-1)
        x_attn = torch.sum(x_attn * mask, dim=-1)
        return x_attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@FEEDFORWARD_NETWORK.register_module()
class ECA_MLP(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 linear=False,
                 drop_path=0.,
                 mlp_hidden_dim=2048,
                 drop=0,
                 eca_drop=0.,
                 norm_layer=nn.LayerNorm,
                 cha_sr_ratio=1,
                 c_head_num=None,
                 eta=1.0):
        super().__init__()
        assert embed_dims % num_heads == 0, f"dim {embed_dims} should be divided by num_heads {num_heads}."

        self.dim = embed_dims
        num_heads = c_head_num or num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gamma = nn.Parameter(eta * torch.ones(embed_dims), requires_grad=True)
        self.cha_sr_ratio = cha_sr_ratio if num_heads > 1 else 1

        # config of mlp for v processing
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.eca_drop = DropPath(eca_drop) if eca_drop > 0. else nn.Identity()
        self.mlp_v = Mlp(in_features=embed_dims // self.cha_sr_ratio, hidden_features=mlp_hidden_dim, drop=drop, linear=linear)
        self.norm_v = norm_layer(embed_dims // self.cha_sr_ratio)
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _gen_attn(self, q, k):
        q = q.softmax(-2).transpose(-1, -2)
        _, _, N, _ = k.shape
        k = torch.nn.functional.adaptive_avg_pool2d(k.softmax(-2), (N, 1))

        attn = torch.nn.functional.sigmoid(q @ k)
        return attn * self.temperature

    def forward(self, x, atten=None):
        x = x.permute(1, 0, 2)
        B, N, C = x.shape
        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = self._gen_attn(q, k)
        attn = self.attn_drop(attn)
        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd * Cv))).reshape(Bv, Nv, Hd, Cv).transpose(
            1, 2)

        repeat_time = N // attn.shape[-1]
        attn = attn.repeat_interleave(repeat_time, dim=-1) if attn.shape[-1] > 1 else attn
        x_new = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        x = x + self.eca_drop(self.gamma * x_new)
        x = x.permute(1, 0, 2)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

@TRANSFORMER_LAYER.register_module()
class FANDetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super(FANDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=dict(
                type='ECA_MLP',
                embed_dims=256,
                num_heads=8,
                qkv_bias=False,
                attn_drop=0.,
                linear=False,
                drop_path=ffn_dropout,
                mlp_hidden_dim=feedforward_channels,
                drop=0,
                eca_drop=0.,
                norm_layer=nn.LayerNorm,
                cha_sr_ratio=1,
                c_head_num=None,
                eta=1.0,
            ),
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])


@TRANSFORMER_LAYER.register_module()
class FANTransformerLayer(BaseTransformerLayer):

    def __init__(self, 
		 attn_cfgs,
                 feedforward_channels,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 ffn_dropout=None,
                 **kwargs):
        super(FANTransformerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=dict(type='ECA_MLP',
                embed_dims=256,
                mlp_hidden_dim=feedforward_channels,),
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            batch_first=batch_first,
            **kwargs)


@TRANSFORMER_LAYER.register_module()
class TAPFANDetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 num_feature_levels=4,
                 **kwargs):
        super(TAPFANDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=dict(
                type='ECA_MLP',
                embed_dims=256,
                num_heads=8,
                qkv_bias=False,
                attn_drop=0.,
                linear=False,
                drop_path=ffn_dropout,
                mlp_hidden_dim=feedforward_channels,
                drop=0,
                eca_drop=0.,
                norm_layer=nn.LayerNorm,
                cha_sr_ratio=1,
                c_head_num=None,
                eta=1.0,
            ),
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.taps = nn.ModuleList()
        self.norm = nn.LayerNorm(256)
        self.norms = nn.ModuleList()
        for i in range(num_feature_levels):
            tap = TAP(embed_dims=256, num_scales=4)
            self.taps.append(tap)
            norm = nn.LayerNorm(256)
            self.norms.append(norm)


    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                spatial_shapes=None,
                **kwargs):
        N, B, C = value.shape
        value = value.permute(1, 0, 2)
        split_sizes = [spatial_shape[0] * spatial_shape[1] for spatial_shape in spatial_shapes]
        split_feats = torch.split(value, split_sizes, dim=1)
        feats = []
        for idx, (split_feat, spatial_shape) in enumerate(zip(split_feats, spatial_shapes)):
            original_feat = split_feat.transpose(1, 2).view(B, C, spatial_shape[0], spatial_shape[1])
            feat = self.taps[idx](original_feat)
            feat = feat.permute(0,2,3,1).reshape(B, -1, C)
            feat = self.norms[idx](feat)
            feats.append(feat)
        value = torch.cat(feats, dim=1)
        value = self.norm(value).permute(1, 0, 2)
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

@TRANSFORMER_LAYER.register_module()
class TAPFANTransformerLayer(BaseTransformerLayer):
    def __init__(self,
		 attn_cfgs,
                 feedforward_channels,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 num_feature_levels=4,
                 **kwargs):
        super(TAPFANTransformerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            ffn_cfgs=dict(type='ECA_MLP',
                embed_dims=256,
                num_heads=8,
                qkv_bias=False,
                attn_drop=0.,
                linear=False,
                mlp_hidden_dim=feedforward_channels,
                drop=0,
                eca_drop=0.,
                norm_layer=nn.LayerNorm,
                cha_sr_ratio=1,
                c_head_num=None,
                eta=1.0,), 
            operation_order=operation_order,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            batch_first=batch_first,
            **kwargs)
        self.taps = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm = nn.LayerNorm(256)
        for i in range(num_feature_levels):
            tap = TAP(embed_dims=256, num_scales=4)
            self.taps.append(tap)
            norm = nn.LayerNorm(256)
            self.norms.append(norm)
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                spatial_shapes=None,
                **kwargs):
        N, B, C = query.shape
        query = query.permute(1, 0, 2)
        split_sizes = [spatial_shape[0] * spatial_shape[1] for spatial_shape in spatial_shapes]
        split_feats = torch.split(query, split_sizes, dim=1)
        feats = []
        for idx, (split_feat, spatial_shape) in enumerate(zip(split_feats, spatial_shapes)):
            original_feat = split_feat.transpose(1, 2).view(B, C, spatial_shape[0], spatial_shape[1])
            feat = self.taps[idx](original_feat)
            feat = feat.permute(0,2,3,1).reshape(B, -1, C)
            feat = self.norms[idx](feat)
            feats.append(feat)
        query = torch.cat(feats, dim=1)
        query = self.norm(query).permute(1, 0, 2)
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

@TRANSFORMER_LAYER.register_module()
class TAPDetrTransformerDecoderLayer(BaseTransformerLayer):
    def __init__(self,
            attn_cfgs,
            feedforward_channels,
            ffn_dropout=0.0,
            operation_order=None,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            ffn_num_fcs=2,
            num_feature_levels=4,
            **kwargs):
        super(TAPDetrTransformerDecoderLayer, self).__init__(
                attn_cfgs=attn_cfgs,
                feedforward_channels=feedforward_channels,
                ffn_dropout=ffn_dropout,
                operation_order=operation_order,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                ffn_num_fcs=ffn_num_fcs,
                **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.taps = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_feature_levels):
            tap = TAP(embed_dims=256, num_scales=2)
            self.taps.append(tap)
            norm = nn.LayerNorm(256)
            self.norms.append(norm)

    
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                spatial_shapes=None,
                **kwargs):
        N, B, C = value.shape
        value = value.permute(1, 0, 2)
        split_sizes = [spatial_shape[0] * spatial_shape[1] for spatial_shape in spatial_shapes]
        split_feats = torch.split(value, split_sizes, dim=1)
        feats = []
        for idx, (split_feat, spatial_shape) in enumerate(zip(split_feats, spatial_shapes)):
            original_feat = split_feat.transpose(1, 2).view(B, C, spatial_shape[0], spatial_shape[1])
            feat = self.taps[idx](original_feat)
            feat = feat.flatten(2).transpose(1, 2)
            feat = self.norms[idx](feat)
            feats.append(feat)
        value = torch.cat(feats, dim=1).permute(1, 0, 2)
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

