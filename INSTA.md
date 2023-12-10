### 空间内核网络部分
#### 空间内核原理
![[Pasted image 20231210135209.png]]
#### 数据集
输入：
 b，c，h，w
 b为样本数目，c为通道数，h为图片高度，w为图片宽度
输出：
 b，c，h，w，k，k
#### 论文原文
![[Pasted image 20231210135552.png]]
#### 过程处理
```python
def __init__(self, c, k):  
    self.SKk = k  
    self.SKconv= nn.Conv2d(c, k ** 2, kernel_size=1) #1*1卷积层
    self.SKbn = nn.BatchNorm2d(k ** 2)  # BN层
```
初始化创建一个1 * 1的卷积层和一个BN层进行归一化
```python
def SKNetwork(self,data):  
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
```
