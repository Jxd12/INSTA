import torch

from DKGenerator import DKGenerator
from CLModule import CLModule

class myModule():
    def __init__(self,way_num,shot_num,episode_size):
        self.DKGenerator = DKGenerator()
        self.CLModule = CLModule()
        self.way_num = way_num
        self.shot_num = shot_num
        self.episode_size = episode_size
    def forward(self,support_data,query_data):
        """
        返回support data,query data经过处理后的特征表示
        Args:
            data:Size为[episode_size,way_num * shot_num, 特征图(大小由backbone决定)]
        Returns:
        """
        support_out = []
        query_out = []
        for i in range(self.episode_size):
            # support set
            # task-adaptive
            taskKernel = self.CLModule.forward(support_data)
            taskKernel = self.DKGenerator.forward(taskKernel)
            # instance-adaptive
            instanceKernel = self.DKGenerator.forward(support_data)
            # Hadamard Product
            dynamicKernel = taskKernel * instanceKernel

            # 卷积
            support_out.append(conv(support_data, dynamicKernel))
            # query data只需要Task-Specific Kernel
            query_out.append(conv(query_data,taskKernel))
        return torch.stack(support_out),torch.stack(query_out)