# RNN 理论
![alt text](image.png)
![alt text](image-1.png)
公式中t和i是一个东西，描述整个过程的平均准确性  
![alt text](image-2.png)
g:所有层的梯度  
如果g的长度大于sita,那么进行投影
![alt text](image-3.png)
one to one 的情况其实就是MLP  
![alt text](image-4.png)

# 总结
理论部分了解概念即可，主要看代码实现