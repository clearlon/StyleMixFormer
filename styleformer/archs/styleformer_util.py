import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from styleformer.archs.arch_util import LayerNorm2d


class EqualLinear(nn.Module):
    """Equalized Linear as StyleGAN2.

    Args:
        in_channels (int): Size of each sample.
        out_channels (int): Size of each output sample.
        bias (bool): If set to ``False``, the layer will not learn an additive
            bias. Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
        lr_mul (float): Learning rate multiplier. Default: 1.
        activation (None | str): The activation after ``linear`` operation.
            Supported: 'fused_lrelu', None. Default: None.
    """

    def __init__(self, in_channels, out_channels, bias=True, bias_init_val=0, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_mul = lr_mul
        self.activation = activation
        if self.activation not in ['fused_lrelu', None]:
            raise ValueError(f'Wrong activation value in EqualLinear: {activation}'
                             "Supported ones are: ['fused_lrelu', None].")
        self.scale = (1 / math.sqrt(in_channels)) * lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels).div_(lr_mul))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if self.bias is None:
            bias = None
        else:
            bias = self.bias * self.lr_mul
        if self.activation == 'fused_lrelu':
            out = F.linear(x, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)
        else:
            out = F.linear(x, self.weight * self.scale, bias=bias)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, bias={self.bias is not None})')

class ModulatedConv2d(nn.Module):
    """Modulated Conv2d used in StyleGAN2.

    There is no bias in ModulatedConv2d.

    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        num_style_feat (int): Channel number of style features.
        demodulate (bool): Whether to demodulate in the conv layer.
            Default: True.
        sample_mode (str | None): Indicating 'upsample', 'downsample' or None.
            Default: None.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. Default: (1, 3, 3, 1).
        eps (float): A value added to the denominator for numerical stability.
            Default: 1e-8.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 groups,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 resample_kernel=(1, 3, 3, 1),
                 eps=1e-8):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        if self.sample_mode == 'upsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=2, downsample_factor=1, kernel_size=kernel_size)
        elif self.sample_mode == 'downsample':
            self.smooth = UpFirDnSmooth(
                resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size)
        elif self.sample_mode is None:
            pass
        else:
            raise ValueError(f'Wrong sample mode {self.sample_mode}, '
                             "supported ones are ['upsample', 'downsample', None].")

        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)
        # modulation inside each modulated conv
        self.modulation = EqualLinear(
            num_style_feat, in_channels, bias=True, bias_init_val=1, lr_mul=1, activation=None)

        self.groups = 1 if groups is None else groups

        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels // self.groups, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        """Forward function.

        Args:
            x (Tensor): Tensor with shape (b, c, h, w).
            style (Tensor): Tensor with shape (b, num_style_feat).

        Returns:
            Tensor: Modulated tensor after convolution.
        """
        b, c, h, w = x.shape  # c = c_in
        # weight modulation
        style = self.modulation(style).view(b, 1, c, 1, 1)
        # self.weight: (1, c_out, c_in, k, k); style: (b, 1, c, 1, 1)
        weight = self.scale * self.weight * style  # (b, c_out, c_in, k, k)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        if self.sample_mode == 'upsample':
            x = x.view(1, b * c, h, w)
            weight = weight.view(b, self.out_channels, c, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(b * c, self.out_channels, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
            out = self.smooth(out)
        elif self.sample_mode == 'downsample':
            x = self.smooth(x)
            x = x.view(1, b * c, *x.shape[2:4])
            out = F.conv2d(x, weight, padding=0, stride=2, groups=b)
            out = out.view(b, self.out_channels, *out.shape[2:4])
        else:
            x = x.view(1, b * c, h, w)
            # weight: (b*c_out, c_in, k, k), groups=b
            out = F.conv2d(x, weight, padding=self.padding, groups=b * self.groups)
            out = out.view(b, self.out_channels, *out.shape[2:4])

        return out

class AdaIN(nn.Module):
    """
    AdaIN follow by IFRnet
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        ch = y.size(1)
        sigma, mu = torch.split(y.unsqueeze(-1).unsqueeze(-1), [ch // 2, ch // 2], dim=1)

        x_mu = x.mean(dim=[2, 3], keepdim=True)
        x_sigma = x.std(dim=[2, 3], keepdim=True)

        return sigma * ((x - x_mu) / x_sigma) + mu


class SpatialInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.spatial_interaction(x)

class ChannelInteraction(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.channel_interaction(x)


##########################################################################
## 
class SimpleGate(nn.Module):
    """
    copy from NAFNet
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class AttentionCC(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(AttentionCC, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, channel_attn):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        v = channel_attn * v

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # out = self.project_out(out)
        return out


class MixRestormer(nn.Module):
    """
    conv mix transformer for image restoration
    """
    def __init__(self,
                 dim, 
                 num_heads, 
                 drop_type='dropout',
                 drop_out_rate=0.                 
                ):
        super().__init__()
        exp_dim = int(dim * 1)

        self.projection1 = nn.Conv2d(dim, exp_dim, 1)
        self.norm1 = LayerNorm2d(exp_dim)
        self.act = nn.GELU()
        
        self.DWConv = nn.Sequential(
            nn.Conv2d(exp_dim // 2, exp_dim // 2, 3, 1, 1),
            nn.GELU()
        )
        self.channel_interaction = ChannelInteraction(exp_dim // 2)
        self.proj_cnn = nn.Conv2d(exp_dim // 2, exp_dim // 2, 1)

        self.attn = AttentionCC(exp_dim // 2, num_heads, bias=True)

        self.spation_attn = SpatialInteraction(dim=exp_dim // 2)

        self.proj_cat = nn.Sequential(
            nn.Conv2d(exp_dim, dim, 1),
            nn.GELU()
        )

        # ffn
        self.norm2 = LayerNorm2d(dim)
        self.ffn1 = nn.Conv2d(dim, dim * 2, 1)
        self.sg = SimpleGate()
        self.ffn2 = nn.Conv2d(dim, dim, 1)

        if drop_type == 'dropout':
            self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
            self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        elif drop_type == 'channel_wise':
            self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
            self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        # b,c,h,w = inp.shape

        x = inp

        x = self.act(self.norm1(self.projection1(x)))
        x_cnn, attn = x.chunk(2, dim=1)
        x_cnn = self.DWConv(x_cnn)
        ch_ia = self.channel_interaction(x_cnn)
        attn = self.attn(attn, ch_ia)
        x_cnn = self.proj_cnn(x_cnn)
        x_cnn = self.spation_attn(attn) * x_cnn
        x = torch.cat([x_cnn, attn], dim=1)
        x = self.proj_cat(x)
        x = self.dropout1(x)

        y = x + inp

        x = self.ffn1(self.norm2(y))
        x = self.sg(x)
        x = self.ffn2(x)
        x = self.dropout2(x)

        return x + y

class StyleFormerBlock(nn.Module):
    """
    modualted by 'mod' and 'demod' follow stylegan2
    """
    def __init__(self, c, 
                    cond_c=6, 
                    drop_type='dropout', drop_out_rate=0.,
                    ):
        super().__init__()
        self.c = c

        self.mixblock = MixRestormer(dim=c, num_heads=8, drop_type=drop_type, drop_out_rate=drop_out_rate)
        self.gamma = nn.Linear(cond_c, c)
        self.modulated1 = ModulatedConv2d(c, c, 3, groups=1, num_style_feat=c, demodulate=True)
        self.act = nn.GELU()

    def forward(self, inp, vector):
        inp = self.mixblock(inp)
        x = inp
        x = self.modulated1(x, self.gamma(vector))
        x = self.act(x + inp)
        return x