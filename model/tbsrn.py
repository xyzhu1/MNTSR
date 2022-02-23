import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed
import warnings
import math, copy

warnings.filterwarnings("ignore")

from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class FeatureEnhancer(nn.Module):

    def __init__(self):
        super(FeatureEnhancer, self).__init__()

        self.multihead = MultiHeadedAttention(h=4, d_model=128, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=128)

        self.pff = PositionwiseFeedForward(128, 128)
        self.mul_layernorm3 = LayerNorm(features=128)

        self.linear = nn.Linear(128, 64)

    def forward(self, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''
        batch = conv_feature.shape[0]
        position2d = positionalencoding2d(64, 16, 64).float().cuda().unsqueeze(0).view(1, 64, 1024)
        position2d = position2d.repeat(batch, 1, 1)
        conv_feature = torch.cat([conv_feature, position2d], 1)  # batch, 128(64+64), 32, 128
        result = conv_feature.permute(0, 2, 1).contiguous()
        origin_result = result
        result = self.mul_layernorm1(origin_result + self.multihead(result, result, result, mask=None)[0])
        origin_result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))
        result = self.linear(result)
        return result.permute(0, 2, 1).contiguous()


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x), attention_map


def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TBSRN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=True, srb_nums=5, mask=False, hidden_units=32,
                 input_channel=3):
        super(TBSRN, self).__init__()

        # self.conv = nn.Conv2d(input_channel, 3,3,1,1)
        # self.bn = nn.BatchNorm2d(3)
        # self.relu = nn.ReLU()

        in_planes = 3
        if mask:
            in_planes = 4

        self.zxyblock1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()

        )
        self.zxyblock2 = RecurrentResidualBlock(2 * hidden_units)
        self.zxyblock3 = RecurrentResidualBlock(2 * hidden_units)
        self.zxyblock4 = RecurrentResidualBlock(2 * hidden_units)
        self.zxyblock5 = RecurrentResidualBlock(2 * hidden_units)
        self.zxyblock6 = RecurrentResidualBlock(2 * hidden_units)

        self.zxyblock7 = nn.Sequential(
            nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * hidden_units)
        )

        block_ = [UpsampleBLock(2 * hidden_units, 2), nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4)]

        self.zxyblock8 = nn.Sequential(*block_)

        self.memory = _Memory_Block(64, 128, 0.999)

        #######下面是原文中的对其方法
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):  ### x:torch.Size([16, 3, 16, 64])

        if self.stn and self.training:  # 这一堆乱七八糟的操作有必要吗
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)  ### x:torch.Size([16, 3, 16, 64])

        x1 = self.zxyblock1(x)

        x2 = self.zxyblock2(x1)
        x3 = self.zxyblock3(x2)
        x4 = self.zxyblock4(x3)
        x5 = self.zxyblock5(x4)
        x6 = self.zxyblock6(x5)

        x6,_=self.memory(x6)

        x7 = self.zxyblock7(x6)
        x8 = self.zxyblock8(x1 + x7)

        return torch.tanh(x8)


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):#x torch.Size([16, 64, 16, 64])

        y = self.avg_pool(x)#   torch.Size([16, 64, 1, 1])
        y = self.conv_du(y)#    torch.Size([16, 64, 1, 1])

        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat=64, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=9, kernel_size=3, reduction=4,
                 bias=False):
        super(ORSNet, self).__init__()

        act = nn.PReLU()
        self.orb1 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat, kernel_size, reduction, act, bias, num_cab)


    def forward(self, x):
        x = self.orb1(x)

        x = self.orb2(x)

        x = self.orb3(x)


        return x


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.gru2 = GruBlock(channels, channels)
        self.zxyorsnet = ORSNet()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # size = residual.shape
        # residual = residual.view(size[0],size[1],-1)
        # residual = self.feature_enhancer(residual)
        # residual = residual.resize(size[0], size[1], size[2], size[3])

        residual = self.zxyorsnet(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x: b, c, w, h
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # b, w, h, c
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])  # b*w, h, c
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class _Memory_Block(nn.Module):
    def __init__(self, hdim, kdim, moving_average_rate=0.999):
        super().__init__()

        self.c = hdim
        self.k = kdim

        self.moving_average_rate = moving_average_rate

        self.units = nn.Embedding(kdim, hdim)

    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''
        if m is None:
            m = self.units.weight.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1]  # (n, ) torch.Size([16384])   返回每一行最大值的索引  文中的公式（2）

        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype)  # (n, k) torch.Size([16384, 512])    这地方实际上是一个0,1的矩阵


        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot  # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data

    def forward(self, x, update_flag=True):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''

        b, c, h, w = x.size()
        assert c == self.c
        k, c = self.k, self.c

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c)  # (n, c)

        m = self.units.weight.data  # (k, c)

        xn = F.normalize(x, dim=1)  # (n, c)
        mn = F.normalize(m, dim=1)  # (k, c)
        score = torch.matmul(xn, mn.t())  # (n, k)


        for i in range(3):
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1)  # (k, c)
            score = torch.matmul(xn, mn.t())  # (n, k)

        soft_label = F.softmax(score, dim=1)
        out = torch.matmul(soft_label, m)  # (n, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return out, score
