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


    def SKNetwork(self,data):
        """
        Spatial Kernel Network 何 TODO
        Args:
            data:
             b, c, h, w
             b为样本总数
             c为通道数
             h为高度
             w为宽度
        Returns:

        """
        batch_size, in_channels, height, width = data.size()
        Gsp = self.SKconv(data)
        Gsp = self.SKbn(Gsp)
        Gsp = Gsp.view(batch_size, height, width, self.SKk, self.SKk)
        Gsp = Gsp.unsqueeze(1)
        Gsp = Gsp.expand(batch_size, in_channels, *Gsp.shape[2:])

        return Gsp