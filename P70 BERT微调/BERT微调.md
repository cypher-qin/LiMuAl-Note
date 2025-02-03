# BERT微调
## 微调BERT
BERT对每一个词元返回抽取了上下文信息的特征向量，针对不同的任务使用其中不同的特征  
![alt text](image.png)
## 任务-句子分类
![alt text](image-1.png)
## 任务-命名实体识别
![alt text](image-2.png)
## 任务-回答问题
![alt text](image-3.png)

## 总结
针对任务不同，只需要增加不同的输出层，使用相应的特征。不同的情境下，输入的表示也可能不一样。