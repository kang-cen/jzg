from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform
from model import Model
import torch.nn as nn
import time
import matplotlib.pyplot  as plt


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#数据变换
transform=transform.Compose(
    [
    transform.ToTensor(),
    transform.Resize((28,28)),#(1,28,28) 等下用一个归一化
     ]
)

writer=SummaryWriter(log_dir="./logs")

#数据准备
train_data=torchvision.datasets.MNIST(root="./data",train=True,transform=transform,download=True)
test_data=torchvision.datasets.MNIST(root="./data",train=False,transform=transform,download=True)
len_train=len(train_data)#6w
len_test=len(test_data)#1w

train_loader=DataLoader(dataset=train_data,batch_size=60,shuffle=True)
test_loader=DataLoader(dataset=test_data,batch_size=10000,shuffle=True)

#模型
model=Model()
model.to(device)

#损失函数
loss_fun=nn.CrossEntropyLoss()
loss_fun.to(device)

#优化器
learnrate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learnrate)

#参数设置
train_step=0
loss_list=[]
accuracy_list=[]
start_time=time.perf_counter()
for epoch in range(30):
    loss_total=0.0
    model.train()#P29_3.24 土堆
    for train_loader_data in train_loader:
        imgs,targets=train_loader_data
        imgs=imgs.to(device)
        targets=targets.to(device)
        
        #梯度清零
        optimizer.zero_grad()
        
        #正向传播
        ouputs=model(imgs)
        
        #损失计算、反向传播、参数更新
        loss=loss_fun(ouputs,targets)
        loss.backward()
        optimizer.step()
        
        loss_total+=loss.item()
        
        if train_step%1000==999:
            writer.add_scalar("loss_epoch",loss_total,train_step)
            print(f"------epoch:{epoch},loss:{loss_total}---------")
        train_step+=1
    
    model.eval()
    for test_loader_data in test_loader:#每轮训练之后都要进行检验精确度
        with torch.no_grad():
            imgs,targets=test_loader_data
            imgs=imgs.to(device)
            targets=targets.to(device)
            ouputs=model(imgs)
            accuracy=(ouputs.argmax(1)==targets).sum()/len_test
            print(f"------epoch:{epoch},accuracy:{accuracy}------")
            writer.add_scalar("accuracy_epoch",accuracy,epoch)
    accuracy_list.append(accuracy.item())
    loss_list.append(loss_total)#将每一轮的loss值记录

print(f"-----模型训练总耗时,cost time:{time.perf_counter()-start_time}s-----")
#绘制曲线    
plt.clf()
plt.figure(1)
plt.plot(loss_list,label="loss_epoch")
plt.legend(loc="best")
plt.title("loss curve")
plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure(2)
plt.plot(accuracy_list,label="accuracy_epoch")
plt.legend(loc="best")
plt.title("accuracy curve")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

writer.close()

#模型保存
print('Finished Training')            
torch.save(model.state_dict(),"basci_dataset/model2.pth") 