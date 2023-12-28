# INSTA
机器学习大作业

我们用同样的config和预训练模型训练两个模型：模型一不使用INSTA，模型二使用INSTA，只要模型二的精度高于模型一精度的4%就成功了。

config：ProtoNet-miniImageNet--ravi-resnet12-5-1-Table2
预训练模型：emb_func_best.pth

config中data_root表示数据集的路径，pretrain_path表示预训练模型的路径，需要自己指定。
config中需要调整的属性都用#TODO标记了,可以快速查找.
## 一.模型一：没有INSTA
用提供的config，调整下面的属性即可：
backbone:
  kwargs:
    avg_pool: true 

## 二.模型二:使用INSTA
1. 将INSTA文件夹放在libfewshot根目录下
2. 使用这里提供的proto_net.py替换./core/model/metric/proto_net.py文件
3. 调整config的属性
backbone:
  kwargs:
    avg_pool: false