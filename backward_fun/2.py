#1.py 是标量求gradient 2.py中涉及到 2d-tensor

"""
`backward()`函数默认情况下只能处理标量张量的梯度计算。
如果您想要计算非标量张量的梯度，可以将`backward()`函数中
传入的参数`gradient`设置为与`f`具有相同形状的张量，
用来指定对应梯度的权重。或者您可以使用`torch.autograd.grad`
函数来明确地计算非标量张量的梯度。    
    
  Only Tensors of floating point and complex dtype can require gradients   
这个错误是由于PyTorch中的张量要求必须是浮点类型或复数类型才能设置`requires_grad=True`，
以便进行自动求导。您可以将张量的数据类型转换为浮点类型后再设置`requires_grad=True`。

"""
import torch as t

if t.cuda.is_available:
    device=t.device("cuda:0")
x=t.randn((3,5),device=device,requires_grad=True)
y=t.randn((5,3),device=device,requires_grad=True)
z=t.randn((3,3),device=device,requires_grad=True)

g=t.matmul(x,y)
f=t.matmul(g,z)

# 进行反向传播，gradient参数需要有与f相同的形状
gradient = t.randn_like(f) 
f.backward(gradient=gradient)

print(x.grad,y.grad,z.grad,z.grad.shape)