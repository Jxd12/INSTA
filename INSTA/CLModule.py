import torch.nn as nn

class CLModule(nn.Module):
   def __init__(self,c):
      super().__init__()
      self.CLM1 = nn.Sequential(
         nn.Conv2d(c,c*2,1),
         nn.BatchNorm2d(c*2),
         nn.ReLU(),
         nn.Conv2d(c*2,c*2,1),
         nn.BatchNorm2d(c*2),
         nn.ReLU()
      )
      self.CLM2 = nn.Sequential(
         nn.Conv2d(c*2,c*2,1),
         nn.BatchNorm2d(c*2),
         nn.ReLU(),
         nn.Conv2d(c*2,c,1),
         nn.BatchNorm2d(c),
         nn.Sigmoid() #激活函数sigmoid
      )
   def CLM_forward(self,featuremap): #NxK,c,h,w
      """
             task specific representation,作为动态核task-adaptive的输入 范 TODO
             Args:
                 data:(NxK),c,h,w
                 输入一个四维的Tensor
                 NxK为样本总数
                 c为通道数
             Returns:
                 返回一个1,c,h,w的四维张量，第一维可以去掉
      """
      map = featuremap
      adap = self.CLM1(map)
      inter = adap.sum(dim=0)
      adap_1 = self.CLM2(inter.unsqueeze(0))
      return adap_1

