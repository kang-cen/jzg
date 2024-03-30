import torch
from model import Model
import cv2 as cv
from PIL import Image
import torchvision.transforms as transform

#只能预测黑色背景 白色数字的灰色（28*28）图片
def main():
    transforms=transform.Compose(
    [
        transform.ToTensor(),
        transform.Resize((28,28)),
        transform.Grayscale(),
     ]
    )

    classes=(0,1,2,3,4,5,6,7,8,9)

    image=Image.open("C:\\Users\\33088\\Pictures\\Camera Roll\\2.jpg")
    image=transforms(image)
    image=image.cuda()
    print(image.size())


    model=Model()
    model.load_state_dict(torch.load("basci_dataset/model.pth"))
    model.cuda()
    model.eval()#对于Dropout 可能会有用
    with torch.no_grad():
        output=model(image)
        index=output.argmax(axis=1)#获取最大值的列表
    print(classes[index])
    

def export_onnx():#"Exports a model into ONNX format.
    model=Model()
    model.load_state_dict(torch.load("basci_dataset/model.pth"))
    model.eval()
    
    input=torch.randn([1,1,28,28])
    torch.onnx.export(model,input,"./basci_dataset/cnn_minst.onnx",verbose=True)
    
if __name__=="__main__":
    main()
    export_onnx()