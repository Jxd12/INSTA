import torch
import torch.nn as nn
from INSTA.fcanet import MultiSpectralAttentionLayer

class DKGenerator(nn.Module):
    # 去掉传入参数args
    def __init__(self, c, spatial_size, sigma, k,device):
        super().__init__()
        self.channel = c
        self.h1 = sigma
        self.h2 = k ** 2
        self.k = k
        self.SKk = k
        self.spatial_size = spatial_size
        self.SKconv= nn.Conv2d(c, k ** 2, kernel_size=1)
        self.SKbn = nn.BatchNorm2d(spatial_size ** 2)  # BN层
        self.CKbn = nn.BatchNorm2d(self.channel)
        c2wh = dict([(512, 11), (640, self.spatial_size)])
        self.channel_att = MultiSpectralAttentionLayer(c, c2wh[c], c2wh[c], sigma=self.h1, k=self.k,device = device, freq_sel_method='low16').to(device)


        # self.args = args

    def forward(self,data):
        """
        生成动态核
        Args:
            data:[N*K,c,h,w]

        Returns:

        """
        CKKernel = self.CKNetwork(data)
        SKKernel = self.SKNetwork(data)
        # harmand product
        return CKKernel * SKKernel


    def CKNetwork(self,data):
        """
        Channel Kernel Network 刘 TODO
        Args:
            data:

        Returns:[b,c,h,w,k,k]

        """
        channel_kernel = self.channel_att(data)
        channel_kernel = self.CKbn(channel_kernel)
        channel_kernel = channel_kernel.flatten(-2)
        channel_kernel = channel_kernel.squeeze().view(channel_kernel.shape[0], self.channel, -1)
        return channel_kernel.unsqueeze(-2)

    def SKNetwork(self, data):
        """
        Spatial Kernel Network
        Args:        data:         b, c, h, w
            b为样本总数
             c为通道数
             h为高度
             w为宽度
        Returns:         b, c, h, w, k, k
             b为样本总数
             c为通道数
             h为高度
             w为宽度
             k为参数
        """
        batch_size, in_channels, height, width = data.size()
        # 提取b，c，h，w
        Gsp = self.SKconv(data)
        # 通过1*1卷积层，将通道数变为k**2
        Gsp = Gsp.flatten(-2).transpose(-1, -2)
        size = Gsp.size()
        # 对Gsp进行归一化处理
        Gsp = Gsp.view(size[0], -1, self.SKk, self.SKk)
        Gsp = self.SKbn(Gsp)
        # 对Gsp进行变形，变成h，w，k，k
        Gsp = Gsp.flatten(-2)
        # 对Gsp进行广播操作，生成c，h，w，k，k
        return Gsp.unsqueeze(-3)