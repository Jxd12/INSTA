import torch
import torch.nn as nn
class DKGenerator():
    def __init__(self, c, k):
        self.SKk = k
        self.SKconv= nn.Conv2d(c, k ** 2, kernel_size=1)
        self.SKbn = nn.BatchNorm2d(k ** 2)  # BN层

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
             b, c, h, w, k, k
             b为样本总数
             c为通道数
             h为高度
             w为宽度

        """
        batch_size, in_channels, height, width = data.size()
        Gsp = self.SKconv(data)
        Gsp = self.SKbn(Gsp)
        Gsp = Gsp.view(batch_size, height, width, self.SKk, self.SKk)
        Gsp = Gsp.unsqueeze(1)
        Gsp = Gsp.expand(batch_size, in_channels, *Gsp.shape[2:])
        return Gsp

if __name__ == '__main__':
    # 创建DKGenerator实例
    generator = DKGenerator(c=3, k=2)

    # 创建输入数据
    data = torch.randn(1, 3, 2, 2)  # 4个样本，3个通道，32x32的图片

    # 运行SKNetwork方法
    output = generator.SKNetwork(data)

    # 打印输出的形状
    print(output.shape)