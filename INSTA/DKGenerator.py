import torch
import torch.nn as nn
class DKGenerator():
    def __init__(self, c, k):
        self.SKk = k
        self.SKconv= nn.Conv2d(c, k ** 2, kernel_size=1)
        self.SKbn = nn.BatchNorm2d(k ** 2)  # BN层

    def forward(self,data):
        """
        生成动态核
        Args:
            data:

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

        Returns:

        """

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
        Gsp = self.SKbn(Gsp)
        # 对Gsp进行归一化处理
        Gsp = Gsp.view(batch_size, height, width, self.SKk, self.SKk)
        # 对Gsp进行变形，变成h，w，k，k
        Gsp = Gsp.unsqueeze(1)
        Gsp = Gsp.expand(batch_size, in_channels, *Gsp.shape[2:])
        # 对Gsp进行广播操作，生成c，h，w，k，k
        return Gsp