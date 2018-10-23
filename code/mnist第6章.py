
# coding: utf-8

# In[3]:


#$6.1-$6.3
#导入Keras及相关模型
import numpy as np
import pandas as pd
from keras.utils import np_utils#导入keras.utils 
np.random.seed(10)#产生随机数据


# In[4]:


#读取mnist数据
from keras.datasets import mnist#导入Keras模块


# In[5]:


(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()#下载或读取mnist数据


# In[6]:


#查看mnist数据
print('train data=',len(x_train_image))
print('test data=',len(x_test_image))


# In[9]:


#查看训练数据
print('(x_train_image:',x_train_image.shape)
print('(y_train_label:',y_train_label.shape)


# In[10]:


import matplotlib.pyplot as plt#导入matplotlib.pyplot模块
def plot_image(image):#定义plot_image函数显示数字图像
    #image为数字图像
    fig=plt.gcf()#设置显示图形的大小
    fig.set_size_inches(2,2)
    plt.imshow(image,cmap='binary')#用plt.imshow显示图形，传入参数image是28*28的图形并以黑白灰度显示
    plt.show#开始绘图


# In[11]:


plot_image(x_train_image[0])#调用plot_image显示训练数据集的第0个数字图像


# In[12]:


y_train_label[0])#查看第0项label数据


# In[13]:


import matplotlib.pyplot as plt#导入matplotlib.pyplot模块
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):#定义 plot_images_labels_prediction函数显示多项mnist数据的images与label。
    #images 数字图像
    #labels 真实值
    #prediction 预测结果
    #idx 开始显示的数据index
    #num 要显示的数据项数，默认值为10，最大值为25
    fig=plt.gcf()#设置显示图形的大小
    fig.set_size_inches(12,14)
    if num>25:num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)#建立子图形为5行5列
        ax.imshow(images[idx],cmap='binary')#显示子图形
        title="label="+str(labels[idx])#设置子图形的title
        if len(prediction)>0:#如果传入了预测结果
            title+=",predict="+str(prediction[idx])#title中加入预测结果
            
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])#不显示刻度
        idx+=1#读取下一项
    plt.show()


# In[14]:


plot_images_labels_prediction(x_train_image,y_train_label,[],0,10)#查看训练数据的前10项数据


# In[15]:


#查看测试数据项数
print('x_test_image:',x_test_image.shape)
print('y_test_label:',y_test_label.shape)


# In[16]:


plot_images_labels_prediction(x_test_image,y_test_label,[],0,10)#查看测试数据的前10项数据


# In[17]:


#$6.4-$6.6
x_Train=x_train_image.reshape(60000,784).astype('float32')#将原来28*28的二维数字图像以reshape转换为一维向量，再以astype转换为float
x_Test=x_test_image.reshape(10000,784).astype('float32')


# In[19]:


print('x_train:',x_Train.shape)#查看每一个数字图像为784个浮点数
print('x_test:',x_Test.shape)


# In[20]:


x_train_image[0]#查看第0个图像转换为浮点数后的内容


# In[21]:


#将浮点型数字图像的数字标准化
x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255


# In[22]:


x_Train_normalize[0]#查看标准化后的结果


# In[23]:


y_train_label[:5]#查看转换为浮点型前的label标签字段


# In[24]:


#label标签字段进行one-hot encoding转换
y_TrainOneHot=np_utils.to_categorical(y_train_label)
y_TestOneHot=np_utils.to_categorical(y_test_label)


# In[25]:


y_TrainOneHot[:5]#查看前5项数据转换后结果

