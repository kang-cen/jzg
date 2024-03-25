import numpy as np
import cv2 as cv
#读取并转换成灰度图像
def read_image(src):
    image=cv.imread(src)
    if image is None:
        print("图像未读取")
        
    blob=np.float32(image)/255.0#归一化
    print(blob.shape,blob.dtype)
    # image2=cv.resize(image,(1000,100))#改变图像宽高
    # print(image2.shape)
    cv.namedWindow("blob",cv.WINDOW_FREERATIO)
    #opencv中可以用float显示，但要保证数据在（0，1）之间
    cv.imshow("blob",blob)
    return blob
    
def read_video(path):
    # cv.VideoCapture()`函数来读取视频文件，
    # `cap.read()`函数来读取视频的一帧，并将返回值分配给`ret`和`frame`两个变量。
    # `ret`是一个标志，表示当前帧是否成功读取，如果成功读取则为`True`，否则为`False`。
    # `frame`则是表示当前帧的图像数据。
    cap=cv.VideoCapture(path)
    while True:
        ret,frame=cap.read()
        if ret is not True:#视频出错或者是最后一帧
            break
        if cv.waitKey(10)==27:
            break
        cv.namedWindow("frame",cv.WINDOW_FREERATIO)
        cv.imshow("frame",frame)
        cv.waitKey(1)#1ms播放一帧 
    cv.destroyAllWindows()
        
    
        

def convertcolor(image,flag):
    if flag=="gray":    
        image_gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        cv.namedWindow("gray",cv.WINDOW_FREERATIO)
        cv.imshow("gray",image_gray)
    elif flag=="hsv":
        image_hsv=cv.cvtColor(image,cv.COLOR_BGR2HSV)
        cv.namedWindow("hsv",cv.WINDOW_FREERATIO)
        cv.imshow("hsv",image_hsv)

#通道分离与合并
def channel_operate(image):
    (b,g,r)=cv.split(image)
    cv.namedWindow("b",cv.WINDOW_FREERATIO)
    cv.namedWindow("g",cv.WINDOW_FREERATIO)
    cv.namedWindow("r",cv.WINDOW_FREERATIO)
    cv.imshow("b",b)
    cv.imshow("g",g)
    cv.imshow("r",r)
    g[:]=0
    r[:]=0
    blue=cv.merge((b,g,r))
    cv.imshow("blue",blue)

#显示自定义图像,显示空白图像
def handcraft_image(image):
    selfimage=np.array(np.random.randint(0,256,(1000,2000,3),dtype=np.uint8))
    cv.imshow("selfimage",selfimage)

    blank=np.ones_like(image)
    blank[:] = 255
    cv.imshow("blank",blank)

def on_mouse(event,x,y,flags,param):
    global startx,starty,endx,endy
    if event==cv.EVENT_LBUTTONDOWN:
        startx=x
        starty=y
        print(f"startx:{startx},starty:{starty}")
        
    elif event==cv.EVENT_MOUSEMOVE:
        if startx>0:
            rect=(startx,starty,x-startx,y-starty)
            color=(0,0,255)
            param=temp.copy()
            cv.rectangle(param,rect,color,10,0)
            cv.imshow("meimei",param)
            
    elif event==cv.EVENT_LBUTTONUP:
        endx=x
        endy=y
        wid=endx-startx
        hei=endy-starty
        print(f"endx:{endx},endy:{endy}")
        print(f"wid:{wid},hei:{hei}")
        if wid>0 and hei>0:
            rect=(startx,starty,wid,hei)
            color=(0,0,255)
            roi=param[starty:starty+hei,startx:startx+wid]
            param=temp.copy()
            cv.rectangle(param,rect,color,10,0)
            cv.imshow("meimei",param)
            cv.namedWindow("roi",cv.WINDOW_FREERATIO)
            cv.imshow("roi",roi)
            startx=-1
            starty=-1
            
if __name__=="__main__":
    src="C:\\Users\\33088\\Pictures\\Camera Roll\\meimei.jpg"
    path="D:\\video\\one.mp4"
    image=read_image(src)#(h,w,c)
    read_video(path)
    #ROI区域提取
    startx=-1
    starty=-1
    endx=-1
    endy=-1
    temp=image.copy()#另外赋值一张图片
    # temp=image  #= 这两张图片其实是一个 只不过起一个别称而已
    # cv.setMouseCallback("meimei",on_mouse,image)
    
    
    cv.waitKey(0)
    cv.destroyAllWindows()
    