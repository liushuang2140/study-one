
# coding: utf-8

# In[1]:


#$7.1-7.6
################################数据预处理
#导入Keras及相关模型
from keras.utils import np_utils#导入keras.utils 
import numpy as np
np.random.seed(10)#产生随机数据


# In[2]:


#读取mnist数据
from keras.datasets import mnist#导入Keras模块
(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()#下载或读取mnist数据


# In[3]:


#将原来28*28的二维数字图像以reshape转换为一维向量，再以astype转换为float
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Test=x_test_image.reshape(10000,784).astype('float32')


# In[4]:


#将浮点型数字图像的数字标准化
x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255


# In[5]:


#label标签字段进行one-hot encoding转换
y_Train_OneHot=np_utils.to_categorical(y_train_label)
y_Test_OneHot=np_utils.to_categorical(y_test_label)


# In[7]:


################################建立模型
#导入所需模型
from keras.models import Sequential
from keras.layers import Dense


# In[8]:


#建立Sequential模型
model= Sequential()


# In[9]:


model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))#建立输入层与隐藏层
#units=256 定义隐藏层神经元个数为256
#input_dim=784 设置输入层神经元个数为784
#kernel_initializer='normal' 使用normal distribution 正态分布的随机数来初始化权重与偏差
#activation='relu' 定义激活函数为relu


# In[10]:


model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))#建立输出层


# In[12]:


print(model.summary())#查看模型的摘要


# In[13]:


################################进行训练
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])#定义训练方式
#loss 设置损失函数
#optimizer 设置优化器
#metrics 设置评估模型的方式


# In[37]:


train_history=model.fit(x=x_Train_normalize,y=y_Train_OneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)#开始训练
#x features数字图像的特征值
#y label数字图像真实的值
#validation_split 验证数据所占的比例
#epochs 训练周期
#batch_size 每批次数据
#verbose 是否显示训练过程


# In[16]:


import matplotlib.pyplot as plt#导入matplotlib.pyplot模块
def show_train_history(train_history,train,validation):#定义show_train_history函数显示训练过程
    #train_history 开始训练中产生的结果
    #train 训练数据的执行结果
    #validation 验证数据的执行结果
    fig=plt.gcf()#设置显示图形的大小
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')#显示图的标题
    plt.ylabel(train)#显示y轴的标签
    plt.xlabel('Epoch')#显示x轴的标签
    plt.legend(['train','validation'],loc='upper left')#设置图例显示'train','validation'，位置为左上角
    plt.show()


# In[17]:


show_train_history(train_history,'acc','val_acc')#画出准确率评估执行结果


# In[18]:


show_train_history(train_history,'loss','val_loss')#画出误差执行结果


# In[20]:


################################测试数据评估模型准确率
scores=model.evaluate(x_Test_normalize,y_Test_OneHot)#评估模型准确率
#model.evaluate（x,y） 
#x 测试数据的features
#y 测试数据的label
print()
print('accuracy=',scores[1])#显示准确率


# In[21]:


################################进行预测
prediction=model.predict_classes(x_Test)


# In[23]:


prediction#查看预测结果的前10项数据


# In[25]:


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
plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=340)#显示前10项预测结果


# In[26]:


#$7.7
import pandas as pd#导入pandas模块
pd.crosstab(y_test_label,prediction,rownames=['label'],colnames=['predict'])#使用pd.crosstab建立混淆矩阵：参数依次为测试数据数字图像的真实值，测试数据数字图像的预测结果，行的名称和列的名称。


# In[27]:


df=pd.DataFrame({'label':y_test_label,'predict':prediction})#建立真实值与预测DataFrame
df[:2]


# In[29]:


df[(df.label==5)&(df.predict==3)]#查询真实值是5，但预测值是3的数据


# In[30]:


plot_images_labels_prediction(x_test_image,y_test_label,prediction,idx=340,num=1)#查看第340项数据


# In[31]:


#$7.8
#将隐藏层的256个神经元改为1000个神经元
model=Sequential()
model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))


# In[32]:


print(model.summary())#查看模型摘要


# In[34]:


#开始训练
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train_normalize,y=y_Train_OneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)


# In[35]:


#查看训练过程的准确率
import matplotlib.pyplot as plt
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.show()
show_train_history(train_history,'acc','val_acc')


# In[36]:


#预测准确率
scores=model.evaluate(x_Test_normalize,y_Test_OneHot)
print()
print('accuracy=',scores[1])

