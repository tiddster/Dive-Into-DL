"""
自动求梯度：
PyTorch提供的autograd包能够根据输入和前向传播过程自动构建计算图，并执行反向传播。
"""
import torch

# 设置requires_grad = True
a = torch.ones(2,2, requires_grad=True)
print(a)
print(a.grad_fn)

b = a + 2
print(b)
print(b.grad_fn)

# mean()是求平均
c = b * b * 3
out = c.mean()
print(c, out)

out.backward()
print(a.grad)

# 若反向传播对象不是标量，则需要在backward函数中添加同维度权值再计算
# 例如上面的out是一个标量所以可以直接out.backward()
# 若反向传播对象是张量
d = torch.tensor([1.0,2.0,3.0,4.0], requires_grad=True)
e = d * 2
f = e.view(2,2)
print(f)
weight = torch.tensor([1,0.1,0.01,0.001], dtype=torch.float).resize_as(f)
f.backward(weight)
print(d.grad)
