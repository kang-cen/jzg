import torch as t 
import numpy as np

if t.cuda.is_available:
    device = t.device("cuda:0")

#`requires_grad`: 是否需要对张量进行自动求导，默认为False。
x=t.randn((1,5),device=device,requires_grad=True)#设置为True可以追踪张量的计算历史，从而实现反向传播。
y=t.randn((5,3),device=device,requires_grad=True)#如果是False 在backward是不会计算相关输入的梯度
z=t.randn((3,1),device=device,requires_grad=True)

print("x:",x,"\n","y:",y,"\n","z:",z)

g=t.matmul(x,y)#矩阵乘法，multiply是数乘  matmul  takes 2 positional arguments
f=t.matmul(g,z) #f是scalar 如果不是的话，需要进行设置gradient: 看2.py
print("g:",g,"\n","f:",f)

#PyTorch中用于计算张量`f`相对于图中所有可学习参数的梯度（导数）的函数。
f.backward()
#向量求导，最后结果要转置。具体可以看书
print("x.grad:",x.grad,"\n","y.grad:",y.grad,"\n","z.grad:",z.grad) 

#下面是手动计算的过程
x_grad=t.matmul(y,z).permute(1,0)#为什么要转置，参考矩阵求导
y_grad=t.matmul(z,x).permute(1,0) #matmul(mat1,mat2)  mat1*mat2  而不是mat2*mat1
z_grad=t.matmul(x,y).permute(1,0)
print(x_grad,y_grad,z_grad)

