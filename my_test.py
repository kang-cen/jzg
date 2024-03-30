import numpy as np
txt_file="D:\\file_set\\dataset\\landmaks\\landmark_output.txt"
lines=[]
with open(txt_file) as read_file:#文件对象赋值给`read_file`。
            for line in read_file:#遍历文件对象`read_file`中的每一行。
                line=line.replace("\n","")
                lines.append(line)
landmarks_frame=lines
contents=landmarks_frame[0].split("\t")
print(contents)
landmarks=np.zeros(10,dtype=np.float32)
for i in range(1,len(contents),2):#对人脸五坐标区域进行获取
            landmarks[i-1]=np.float32(contents[i])    #/w是对其进行归一化
            landmarks[i]=np.float32(contents[i+1])
landmarks=landmarks.astype("float32").reshape(-1,2)
print(landmarks)