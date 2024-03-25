#随便写写
import numpy as np;
a=[1,2,3,5,4,1,2,3]
print(a)
a_array=np.array(a)
a_array.dtype=np.int32
print("what is the a_array",a_array)
print("what is  the shape",a_array.shape)
print("what is  the dimension",a_array.ndim)
print("what is  the size",a_array.size)
print("what is  the itemsize",a_array.itemsize)
print("what is  the type",a_array.dtype)

b=np.array([[1,2,3,4,5],[9,8,7,6,0]])
print("what is the a_array",b)
print("what is  the shape",b.shape)
reb=np.reshape(b,(5,2))
print("what is the a_array",reb)
print("what is  the shape",reb.shape)
reb1=reb.T
print("what is the a_array",reb1)
print("what is  the shape",reb1.shape)
reb2=reb1.transpose()
print("what is the a_array",reb2)
print("what is  the shape",reb2.shape)

c=np.array([[[[1,2,3,4,8,4,55,54]]]])
print("\n",c)
print(c.shape)
c1=np.squeeze(c)#直接全部降维
print(c1.shape)
print(c1)
c2=np.expand_dims(c1,1) #axis=0 在row上面扩展，1在col上面扩展
print(c2.shape)
print(c2)

print(c2[c2.argmax()])#ndarray.argmax()返回最大值对应的下标  or np.argmax(temp,axis=0)  o代表列 1代表行

d=np.zeros((1,3,4,5),dtype=np.int16)
print(d,d.shape)
d1=np.zeros_like(d)
print(d1)

e=np.linspace(0,100,1000,True)
print(e)

e1=np.reshape(e,(1,10,10,10))
print("e1",e1)
print(e1.shape)
e2=e1.squeeze()
print(e2.shape)
e3=e2.squeeze()#未变化 只有维度是一才会变化
print(e3.shape)
e_max=e3.max(axis=1) #在axis=0（列）找最大值，axis=1行找最大值
print(e_max,e_max.shape)

f=np.array([
    [
        [1,2,3,4],
        [2,1,3,4],
        [11,23,44,66],
    ],
    [
        [111,222,333,444],
        [999,888,666,444],
        [121,213,424,626],
    ]
        ])
print(f.shape)
f_max=f.max(axis=0)
print(f_max,f_max.shape)