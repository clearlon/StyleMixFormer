import torch
import torch.nn as nn
from torch.nn import functional as F

from styleformer.utils.registry import ARCH_REGISTRY
from styleformer.archs.ResNetArcFace_arch import ResNetArcFace
from styleformer.archs.styleformer_util import StyleFormerBlock


@ARCH_REGISTRY.register()
class StyleFormerNet(nn.Module):
    # partially dynamic depthwise
    def __init__(self, 
                 img_channel=3, 
                 embedding=32, width=32, 
                 kernel_size=3,
                 enc_blk_nums=[1,1,7,7],
                 middle_blk_num=2,
                 dec_blk_nums=[1,1,1,1],
                 drop_type='channel_wise', drop_out_rate=0.5,
                 load_path='experiments/pretrained/resnetarcface.pth',
                ):
        super().__init__()
        ###########################################################################
        ## embedding network
        self.embedding_generate = ResNetArcFace(img_channel, 'IRBlock', [1,1,1,1], True, embedding, load_path)

        ###########################################################################
        ## backbone
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=kernel_size, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=kernel_size, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.ModuleList(
                    [StyleFormerBlock(chan, cond_c=embedding, drop_type=drop_type, drop_out_rate=drop_out_rate) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.ModuleList(
                    [StyleFormerBlock(chan, cond_c=embedding, drop_type=drop_type, drop_out_rate=drop_out_rate) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),nn.PixelShuffle(2))
            )
            chan = chan // 2
            self.decoders.append(
                nn.ModuleList(
                    [StyleFormerBlock(chan, cond_c=embedding, drop_type=drop_type, drop_out_rate=drop_out_rate) for _ in range(num)])
            )

    def forward(self, inp):
        embedding = self.embedding_generate(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            for sub_enc in encoder:
                x = sub_enc(x, embedding)
            encs.append(x)
            x = down(x)

        for middle_blk in self.middle_blks:
            x = middle_blk(x, embedding)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            for sub_dec in decoder:
                x = sub_dec(x, embedding)

        x = self.ending(x)
        x = x + inp

        return x

