import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(28*28,100)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(100,10)
        
        
    def forward(self,x):
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)#不要softmax是因为交叉熵损失自带 
        return x
    
if __name__=="__main__":
    input=torch.ones((1,28,28))
    model=Model()
    output=model(input)
    print(input.size())
    print(output.size())