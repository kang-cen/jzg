import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import cv2 as cv

class FaceLandmarkDataset(Dataset):
    def __init__(self,txt_file) -> None:
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64,64),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        lines=[]
        with open(txt_file) as read_file:#文件对象赋值给`read_file`。
            for line in read_file:#遍历文件对象`read_file`中的每一行。
                line=line.replace("\n",'')#使用`replace()`函数将末尾的换行符`\n`替换为空字符串，然后将处理后的行添加到列表`lines`中。
                lines.append(line)
        self.landmarks_frame=lines#初始化图片与标签数据
        
    def __len__(self):
        return len(self.landmarks_frame)#返回数据集大小
    
    def num_of_samples(self):
        return len(self.landmarks_frame)
 #魔术方法在类或对象的某些事件出发后会自动执行，如果希望根据自己的程序定制特殊功能的类，
 # 那么就需要对这些方法进行重写。使用这些「魔法方法」，我们可以非常方便地给类添加特殊的功能。   
    def __getitem__(self, index) :
        if torch.is_tensor(index):
            index=index.tolist()
        contents=self.landmarks_frame[index].split("\t")#将他们以tab键分开 contents是一个列表 
        image_path=contents[0] #获取图像路径
        img=cv.imread(image_path)
        h,w,c,=img.shape
        landmarks=np.zeros(10,dtype=np.float32)
        for i in range(1,len(contents),2):#对人脸五坐标区域进行获取
            landmarks[i-1]=np.float32(contents[i])/w    #/w是对其进行归一化
            landmarks[i]=np.float32(contents[i+1])/h
        landmarks=landmarks.astype("float32").reshape(-1,2)#处理后的标签格式
        
        sample={"image":self.transform(img),"landmarks":torch.from_numpy(landmarks)}#from_numpy转化为tensor
        return sample
    
if __name__=="__main__":
    ds=FaceLandmarkDataset("D:\\file_set\\dataset\\landmaks\\landmark_output.txt")
    for i in range(len(ds)):#当执行 len(ds) 时，就会调用 __len__ 方法，这个方法主要用于读取容器内元素的数量
        sample=ds[i]#当执行 my_list[0] 时，就会调用 __getitem__ 方法，这个方法主要用于从容器中读取元素。
        print(i,sample["image"].size(),sample["landmarks".size()])
        if i==3:
            break
        
        dataloader=DataLoader(ds,batch_size=4,shuffle=True)
        for i_batch,sample_batched in enumerate(dataloader):
            print(i_batch,sample_batched["image"].size(),sample_batched["landmarks".size()])
               