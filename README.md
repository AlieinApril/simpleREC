# 一个简单的推荐系统搭建(回归)

# 一、项目背景

试图模仿youtube经典双塔推荐系统架构

# 二、数据集简介

本项目用所采用的的数据集是movielens 1m的数据集(100K)

## 1.数据加载和预处理


```python
import pandas as pd
import numpy as np
movie_data = pd.read_csv("/data/movies.csv",usecols=[0,2])
attrs = []
for index,row in movie_data.iterrows():
    # print(row[1])
    genres = row[1].split("|")
    for genre in genres:
        if genre not in attrs:
            attrs.append(genre)
movie_attrs = []
for row in movie_data["genres"].str.split("|"):
    temp = np.zeros(shape=(20))
    for attr in row:
        if attr in attrs:
            temp[attrs.index(attr)] = 1
    movie_attrs.append(temp)
movie_attrs = pd.DataFrame(movie_attrs)
movie_data = pd.concat([movie_data,movie_attrs],axis=1)
movie_data.drop("genres",axis=1,inplace=True)
```




# 三、模型选择和开发


## 1.模型组网

```python
class Rec(paddle.nn.Layer):
    def __init__(self):
        super(Rec, self).__init__()

        self.linear_1 = paddle.nn.Linear(21, 512)
        self.linear_2 = paddle.nn.Linear(512, 256)
        self.linear_3 = paddle.nn.Linear(256, 128)
        self.output= paddle.nn.Linear(128, 1)
        self.relu = paddle.nn.ReLU()
        self.relu_2 = paddle.nn.ReLU()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.relu(y)
        y = self.linear_2(y)
        y = self.relu_2(y)
        y = self.linear_3(y)
        y = self.output(y)
       # y = paddle.clip(y,min=0,max=5)

        return y
```

## 2.模型网络结构可视化


```python
# 模型封装
model = paddle.Model(network)

# 模型可视化
---------------------------------------------------------------------------
 Layer (type)       Input Shape          Output Shape         Param #    
===========================================================================
   Linear-1          [[64, 21]]           [64, 512]           11,264     
    ReLU-1          [[64, 512]]           [64, 512]              0       
   Linear-2         [[64, 512]]           [64, 256]           131,328    
    ReLU-2          [[64, 256]]           [64, 256]              0       
   Linear-3         [[64, 256]]           [64, 128]           32,896     
   Linear-4         [[64, 128]]            [64, 1]              129      
===========================================================================
Total params: 175,617
Trainable params: 175,617
Non-trainable params: 0
---------------------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.81
Params size (MB): 0.67
Estimated Total Size (MB): 1.49
---------------------------------------------------------------------------

{'total_params': 175617, 'trainable_params': 175617}

## 3.模型训练


```python
# 配置优化器、损失函数、评估指标
model_ = paddle.Model(rec)
model_.prepare(optimizer=paddle.optimizer.Adam(parameters=model_.parameters()),
              loss=paddle.nn.MSELoss(reduction="mean"),
              metrics=MSE())
              
# 启动模型全流程训练
model_.fit(custom_dataset,
            epochs=100,
            batch_size=64,
            verbose=1)
```

    Epoch 97/100
step 10/16 [=================>............] - loss: 0.5847 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
step 16/16 [==============================] - loss: 0.7768 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True
       [710.24444580]) - 7ms/step          
Epoch 98/100
step 10/16 [=================>............] - loss: 0.7500 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
step 16/16 [==============================] - loss: 0.6261 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True
       [675.09747314]) - 6ms/step          
Epoch 99/100
step 10/16 [=================>............] - loss: 0.6798 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
step 16/16 [==============================] - loss: 1.0774 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True
       [769.93090820]) - 6ms/step          
Epoch 100/100
step 10/16 [=================>............] - loss: 0.4347 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True,
step 16/16 [==============================] - loss: 0.2434 - MSE: Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=True
       [672.81494141]) - 7ms/step  
    


## 4.模型预测


y_pre = np.array(model_.predict(train_loader.dataset[120:140][0]))
y_true = np.array(train_loader.dataset[120:140][1])
for i in range(20):
    print("预测值:真实值 : ",np.round(y_pre[0,i],1),y_true[i],end="\n")
    
    
 Predict begin...
step 20/20 [==============================] - 4ms/step         
Predict samples: 420
预测值:真实值 :  [4.9] [5.]    <br />
预测值:真实值 :  [4.2] [4.]    <br />
预测值:真实值 :  [5.5] [5.]    <br />
预测值:真实值 :  [4.6] [4.]    <br />
预测值:真实值 :  [5.7] [5.]    <br />
预测值:真实值 :  [4.7] [4.]    <br />
预测值:真实值 :  [4.9] [5.]    <br />
预测值:真实值 :  [5.2] [5.]    <br />
预测值:真实值 :  [4.9] [5.]    <br />
预测值:真实值 :  [4.] [3.]    <br />
预测值:真实值 :  [4.5] [5.]    <br />
预测值:真实值 :  [4.9] [4.]    <br />
预测值:真实值 :  [3.7] [4.]    <br />
预测值:真实值 :  [4.5] [4.]    <br />
预测值:真实值 :  [5.] [5.]    <br />
预测值:真实值 :  [5.5] [5.]    <br />
预测值:真实值 :  [5.5] [5.]    <br />
预测值:真实值 :  [4.9] [5.]    <br />
预测值:真实值 :  [5.3] [5.]    <br />
预测值:真实值 :  [4.5] [4.]    <br />
