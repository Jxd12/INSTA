import torch
import torch.nn as nn

from INSTA.DKGenerator import DKGenerator
from INSTA.CLModule import CLModule

class myModule(nn.Module):
    def __init__(self):
        super(myModule, self).__init__()
        # TODO；config中获得参数
        self.way_num = 5
        self.shot_num = 1
        self.episode_size = 1
        self.channel = 640
        self.imageSize = 5 # 对应h，w的值
        self.sigma = 0.2
        self.k = 3
        self.device = "cuda" # 默认在cuda上运行
        # self.device = "cpu"

        self.DKGenerator = DKGenerator(self.channel,self.imageSize,self.sigma,self.k,device=self.device).to(self.device)
        self.CLModule = CLModule(self.channel).to(self.device)

    def unfold(self, x, padding, k):
        x_padded = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2] + 2 * padding,
                                          x.shape[3] + 2 * padding).fill_(0)
        x_padded[:, :, padding:-padding, padding:-padding] = x
        x_unfolded = torch.cuda.FloatTensor(*x.shape, k, k).fill_(0)
        for i in range(int((self.k + 1) / 2 - 1), x.shape[2] + int((
                                                                           self.k + 1) / 2 - 1)):  ## if the spatial size of the input is 5,5, the sampled index starts from 1 ends with 7,
            for j in range(int((self.k + 1) / 2 - 1), x.shape[3] + int((self.k + 1) / 2 - 1)):
                x_unfolded[:, :, i - int(((self.k + 1) / 2 - 1)), j - int(((self.k + 1) / 2 - 1)), :, :] = x_padded[:,
                                                                                                           :, i - int(
                    ((self.k + 1) / 2 - 1)):i + int((self.k + 1) / 2), j - int(((self.k + 1) / 2 - 1)):j + int(
                    ((self.k + 1) / 2))]
        return x_unfolded

    def forward(self,support_data,query_data):
        """
        返回support data,query data经过处理后的特征表示
        Args:
            data:Size为[episode_size,way_num * shot_num, 特征图(大小由backbone决定)]
        Returns:
        """
        s_size = support_data.size()
        q_size = query_data.size()
        support_data = support_data.reshape(s_size[0], s_size[1], 640, 5, 5)
        query_data = query_data.reshape(q_size[0], q_size[1], 640, 5, 5)


        support_out = []
        query_out = []
        for i in range(self.episode_size):
            # support set
            # # task-adaptive
            support_data = support_data[i]
            query_data = query_data[i]
            # instance-adaptive
            instanceKernel = self.DKGenerator.forward(support_data)
            # Hadamard Product
            instanceKernel_shape = instanceKernel.size()
            feature_shape = support_data.size()
            instanceKernel = instanceKernel.view(instanceKernel_shape[0],instanceKernel_shape[1],feature_shape[-2],feature_shape[-1],self.k,self.k)

            taskKernel = self.CLModule.CLM_forward(support_data)
            taskKernel = self.DKGenerator.forward(taskKernel)
            taskKernel_shape = taskKernel.size()
            taskKernel = taskKernel.view(taskKernel_shape[0], taskKernel_shape[1], feature_shape[-2],
                                           feature_shape[-1], self.k, self.k)

            dynamicKernel = taskKernel * instanceKernel

            # 卷积操作？
            unfold_feature = self.unfold(support_data, int((self.k + 1) / 2 - 1), self.k)  ## self-implemented unfold operation
            adapted_s = (unfold_feature * dynamicKernel).mean(dim=(-1, -2)).squeeze(-1).squeeze(-1) + support_data
            adapted_s = nn.AdaptiveAvgPool2d(1)(adapted_s).squeeze(-1).squeeze(-1)

            support_out.append(adapted_s)

            query_ = nn.AdaptiveAvgPool2d(1)((self.unfold(query_data, int((taskKernel.shape[-1] + 1) / 2 - 1),
                                                                taskKernel.shape[-1]) * taskKernel)).squeeze()
            adapted_q = query_data + query_
            adapted_q = nn.AdaptiveAvgPool2d(1)(adapted_q).squeeze(-1).squeeze(-1)
            query_out.append(adapted_q)

        support_out = torch.stack(support_out)
        query_out = torch.stack(query_out)
        # s_size = support_out.size()
        # q_size = query_out.size()

        # support_out = support_out.view(s_size[0],s_size[1],s_size[-1] * s_size[-2] * s_size[-3])
        # query_out = query_out.view(q_size[0],q_size[1],q_size[-1] * q_size[-2] * q_size[-3])
        return support_out,query_out