import numpy as np
import torch as t


'''
1。'torch. rand（size）'生成一个张量,其中包含从0到1之间的均匀分布中采样的随机数。
'size'参数指定输出张量的形状。

2.'torch. randn（size）'从标准正态分布（均值=0，std=1）生成一个充满随机数的张量。
'size'参数指定输出张量的形状。

3.'torch. randint（low，high，size）'生成一个张量，其中填充了从范围'[low，high）
'均匀采样的随机整数。'low'和'high'参数指定要采样的整数范围，'size'参数指定输出张量的形状。

'''
print(t.__version__)
x=t.empty(4,4)#create a empty tensor  不一定为空 当size过大时
y=t.rand(4,4)
z=t.zeros(10,10)#真正的空张量
m1=t.tensor([1,2,3,4,5,6,7,8,9])
m2=t.tensor([9,8,7,6,5,4,3,2,1])
print(m1+m2)
print(t.add(m1,2))#(arg1,arg2) 可以是矩阵也可以是向量还可以是标量
print(t.subtract(m2,m1))#m2-m1
print(t.subtract(m2,2))#m2-2
print(m1-m2)
print("m1*m2:",m1*m2)#对应位数相乘
print("using multipy:",t.multiply(m1,m2))#点成
print("m2/m1:",m2/m1)
print("using divide:",t.divide(m1,m2))#m1/m2
# print(m1*t.transpose(m2))  #还带开发

#tensor.view() and torch.permute
tensor = t.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
tensor_view=tensor.view(3,4)#-1`表示由PyTorch自动计算该维度的大小，从而保证张量中元素的总数不变
tensor_view1=tensor.view(-1,6)
print("tensor.view(row,col) 其实和reshape一样 都是改变维度的")
print(tensor_view,tensor_view1)
tensor_reshape=t.reshape(tensor,(6,2))
tensor_reshape1=tensor.reshape((-1,4))-1#表示由PyTorch自动计算该维度的大小
# tensor_reshape=tensor.reshape((6,2)) 和上面那个一样 
print(tensor_reshape,tensor_reshape1)
#tensor.permute((1,0))`来交换张量的维度顺序,在这种情况下，`(1,0)`表示将原始张量的维度1和维度0交换位置
tensor_permute=tensor.permute((0,1))#保持不变 原始（0，1，2）——>（x,y,z）
tensor_permute1=tensor.permute((1,0))# 起到转置的过程
print(tensor_permute,tensor_permute1)

#tensor与numpy的转换
t1=t.randn(4,4)
n1=np.random.randn(4,4)
print(t1,"\n",n1)
t12n1=t1.numpy()#将tensor转换成numpy
n12t1=t.from_numpy(n1)#将numpy转换成tensor
n12t2=t.as_tensor(n1)#这个函数也可以实现将numpy转换成tensor
print(t12n1,"\n",n12t1)
print(n12t2)

#将使用GPU计算，将内存放到显存上
if t.cuda.is_available:
    result=m1.cuda()+m2.cuda()
    print(result)
    dd=t.detach(result).cpu().numpy()#先放在内存在解析给numpy格式
    print(dd)

#argmax() and argmin()
if t.cuda.is_available:
    s1=t.tensor([1,2,3,4,5,199])
    print("s1:",s1)
    s1_maxindex=s1.cuda().argmax()
    s1_minindex=t.argmin(s1.cuda())
    print("s1 max:",s1[s1_maxindex],"s1 min:",s1[s1_minindex])

    r1=t.randn(3,3)
    print("r1:",r1)
    r1_maxindex=r1.cuda().argmax()#返回值是标量 tensor形式  但是r1是矩阵  r1[8]肯定会出错啊
    r1_minindex=t.argmin(r1.cuda())#要么将8转换成元组形式 从左到右来 ，要么将r1展平  
    print(r1_minindex,r1_maxindex)
    print("r1 max:",r1.cuda().flatten()[r1_maxindex],"r1 min:",r1.cuda().flatten()[r1_minindex])
