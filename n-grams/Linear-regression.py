#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# In[2]:


x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


# In[3]:


x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# In[5]:


class LinearRegression(nn.Module):
    
    def __init__(self):
        super(LinearRegression , self).__init__()
        self.linear = nn.Linear(1,1) #1 input and 1 output
    
    def forward(self,x):
        out  = self.linear(x)


# In[12]:


model = LinearRegression()
# 定义loss和优化函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


# In[16]:


optimizer 


# In[14]:


epochs = 1000
for epoch in range(epochs):
    inputs = x_train
    target = y_train
    
    #forward
    out = model(inputs)
    loss = criterion(out , target)
    
    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch) %20 ==0:
        print(f"epochs : {epoch}\nloss : {loss.item}:.6f")


# In[ ]:


model.eval()
with torch.no_grad():
    predict = model(x_train)
predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
# 显示图例
plt.legend() 
plt.show()

# 保存模型
torch.save(model.state_dict(), './linear.pth')


# In[ ]:




