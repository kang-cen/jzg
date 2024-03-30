import cv2 as cv

mnist_net=cv.dnn.readNetFromONNX("basci_dataset\cnn_minst.onnx")
image=cv.imread("C:\\Users\\33088\\Pictures\\Camera Roll\\2.jpg")
gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow("input",gray)
#H,W-->N,C,H,W
blob=cv.dnn.blobFromImage(gray,0.00392,(28,28),(127.0))/0.5
print(blob.shape)
mnist_net.setInput(blob)
result=mnist_net.forward()
cv.waitKey(0)
cv.destroyAllWindows()