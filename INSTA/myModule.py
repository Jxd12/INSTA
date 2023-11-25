from DKGenerator import DKGenerator
from CLModule import CLModule

class myModule():
    def __init__(self):
        self.DKGenerator = DKGenerator()
        self.CLModule = CLModule()

    def forward(self,data):
        """
        返回经过动态核处理后的特征表示
        Args:
            data:

        Returns:

        """

        # support set
        # task-adaptive
        taskKernel = self.CLModule.forward(data)
        taskKernel = self.DKGenerator.forward(taskKernel)

        # instance-adaptive
        instanceKernel = self.DKGenerator.forward(data)

        # Hadamard Product
        dynamicKernel = taskKernel * instanceKernel

        # 卷积
        out = conv(data,dynamicKernel)
        return out

        # query set
        # task-adaptive
        taskKernel = self.CLModule.forward(data)
        taskKernel = self.DKGenerator.forward(taskKernel)
        out = conv(data, taskKernel)
        return out

