class DKGenerator():
    def __init__(self):

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

        Returns:

        """