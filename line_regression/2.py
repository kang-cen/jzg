import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#数据准备
input_x=torch.tensor([1,2,0.5,2.5,2.6,3.1]).reshape(-1,1)
input_x=input_x.to(device)
targets=torch.tensor([3.7,4.6,1.65,5.68,5.98,6.95]).reshape(-1,1)
targets=targets.to(device)

class Model(nn.Module):
    def __init__(self,in_dim,out_dim) -> None:
        super().__init__()
        self.linear1=nn.Linear(in_dim,out_dim)
        
    def forward(self,x):
        x=self.linear1(x)
        return x
    
in_dimension=1
out_dimension=1
model=Model(in_dimension,out_dimension)
model.to(device)

loss_fun=nn.MSELoss()
loss_fun.to(device)
learnrate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learnrate) 
loss_list=[]
for epoch in range(100):
    total_loss=0.0
    optimizer.zero_grad()
    outputs=model(input_x)
    loss=loss_fun(outputs,targets)
    loss.backward()
    total_loss+=loss.item()
    optimizer.step()
    if epoch%10==9:
        print(f"-------------epoch:{epoch},total_loss:{total_loss}----------------")
        loss_list.append(total_loss)
               
predicted=model(input_x)
# print(predicted)
plt.figure(1)
plt.plot(loss_list,label="loss value")
plt.legend(loc='best')
plt.figure(2)
plt.plot(input_x.cpu().numpy(),targets.cpu().numpy(),'o',label="true")
plt.plot(input_x.cpu().numpy(),predicted.cpu().detach().numpy() ,"--",label="Prediction")
plt.legend(loc='best')
plt.show()
