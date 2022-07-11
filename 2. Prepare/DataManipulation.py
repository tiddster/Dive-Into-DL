"""
数据操纵：
在PyTorch中，torch.Tensor是存储和变换数据的主要工具。
如果你之前用过NumPy，你会发现Tensor和NumPy的多维数组非常类似。
然而，Tensor提供GPU计算和自动求梯度等更多功能，这些使Tensor更加适合深度学习。
"""
import torch
import numpy as np

"""
创建操作
"""
# 创造一个空数组
a1 = torch.empty(5,3)

# 创造随机数组
xx = torch.rand(5,3)
yy = torch.rand(5,3)

# 根据数据进行创建
a2 = torch.tensor([5.5, 3])

#获取形状
x = a2.shape
x = a2.size()


"""
运算操作
"""
print(xx,yy)

# 加法、指定输出
res = torch.zeros_like(xx)
torch.add(xx,yy, out=res)
print(xx+yy)
print(res)

# 改变形状
zz1 = xx.resize(15)
zz2 = xx.view(-1, 5)
print(zz1)
print(zz2)

# 将标量tensor转换成一个python number
c = torch.rand(1)
print(c.item())

"""
广播机制：
前面我们看到如何对两个形状相同的Tensor做按元素运算。
当对两个形状不同的Tensor按元素运算时，可能会触发广播（broadcasting）机制：
先适当复制元素使这两个Tensor形状相同后再按元素运算
"""
e = torch.arange(1, 3).view(1, 2)
print(e)
f = torch.arange(1, 4).view(3, 1)
print(f)
print(e + f)

"""
tensor转numpy
"""
xxx = xx.numpy()
xxxx = torch.from_numpy(xxx)
print(xxx, xxxx)

"""
tensor On GPU
"""
if torch.cuda.is_available():
    device = torch.device("cuda")
    aa = torch.ones_like(xx, device=device)
    bb = xx.to(device)
    cc = aa + bb
    print(cc)
    print(cc.to(device))
