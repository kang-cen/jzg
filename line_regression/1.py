import torch as t
import numpy as np
import matplotlib.pyplot as plt
#数据准备
input_x=t.tensor([1,2,0.5,2.5,2.6,3.1]).reshape(-1,1)#没有reshape不能进行矩阵乘法
y=t.tensor([3.7,4.6,1.65,5.68,5.98,6.95]).reshape(-1,1)#代码层面中 y=x*W+b 

#模型搭建
class LinearGegression(t.nn.Module):
    def __init__(self,indimension,outdimention):
        # ython3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx :
        super(LinearGegression,self).__init__()
        # Module.__init__(self,)
        # super(Cat, self).__init__(name,age)  # 不要忘记从Animal类引入属性)
        # super().__init__(x, y) #父类中有属性x,y
        
        # 在PyTorch中，当调用`nn.Linear(in_features, out_features)`创建线性模块时，
        # 内部会自动管理权重矩阵w和偏置b，并在计算时应用公式y=x*w + b，而不需要手动转置w。
        # 因此，无需显式转置权重矩阵w。
        self.lieanr=t.nn.Linear(indimension,outdimention)
    def forward(self,x):
        out=self.lieanr(x)
        return out

input_dim=1
output_dim=1#权重矩阵w的形状为(in_features, out_features)  
model=LinearGegression(input_dim,output_dim)

#损失函数
criterion=t.nn.MSELoss()

#优化器
learnrate=0.01
optimizer=t.optim.SGD(model.parameters(),learnrate)

for epoch in range(1000):
    
    #清除梯度
    optimizer.zero_grad()
    
    #正向传播
    # 实际上，当您定义一个继承自`torch.nn.Module`的类来构建模型时，
    # PyTorch会自动调用`forward`方法来定义模型的前向传播过程。因此，
    # 当您调用`model(input_x)`时，实际上是调用了模型类内部的`forward`方法。
    # 这种简洁的写法更为常见和推荐，而且更加直观和简洁。
    # output_y=model.forward(input_x)   
    output_y=model(input_x)
    
    #计算损失
    loss=criterion(output_y,y)
    
    #反向传播--计算各个参数梯度
    loss.backward()
    
    #更新梯度
    optimizer.step()
    
    print("epoch {},loss {}".format(epoch,loss.item()))
    
predicted_y=model(input_x).data.numpy()
print("实际值：",y)
print("预测值：",predicted_y)
    
x=input_x.detach().numpy()#画图的时候不能使用张量
#clear figure
plt.clf()

plt.plot(x,y,"go",label="True data",alpha=0.5)
plt.plot(x,predicted_y,"--",label="Prediction",alpha=0.5)
plt.legend(loc="best")
plt.show()